import asyncio
import contextlib
import json
import logging
import time
from collections.abc import AsyncGenerator
from typing import Any

import mlx.core as mx

from mlx_ollama.engine.model_manager import LoadedModel, ModelManager, parse_keep_alive
from mlx_ollama.config import settings
from mlx_ollama.engine.template_caps import TemplateCaps
from mlx_ollama.utils.streaming import CancellableStream, async_mlx_stream
from mlx_ollama.utils.timing import Timer, TimingStats

logger = logging.getLogger(__name__)

# Serialize inference to prevent concurrent Metal command buffer access
_inference_lock = asyncio.Lock()


@contextlib.contextmanager
def _inference_ref(lm: LoadedModel):
    """Track active inference on a model to prevent expiry during use."""
    lm.active_refs += 1
    try:
        yield
    finally:
        lm.active_refs -= 1
        # Refresh expiry so the model doesn't expire immediately after inference
        ka = parse_keep_alive(settings.default_keep_alive)
        if ka is not None:
            lm.expires_at = time.time() + ka


def _build_generate_kwargs(options: dict | None, is_vlm: bool = False) -> dict:
    """Convert Ollama options dict to mlx_lm/mlx_vlm generate kwargs."""
    if not options:
        return {}
    kwargs = {}
    # mlx-lm uses "temp", mlx-vlm uses "temperature"
    temp_key = "temperature" if is_vlm else "temp"
    mappings = {
        "temperature": temp_key,
        "top_p": "top_p",
        "top_k": "top_k",
        "seed": "seed",
        "num_predict": "max_tokens",
        "repeat_penalty": "repetition_penalty",
        "repeat_last_n": "repetition_context_size",
        "min_p": "min_p",
    }
    for ollama_key, mlx_key in mappings.items():
        if ollama_key in options:
            kwargs[mlx_key] = options[ollama_key]
    # stop is only supported by mlx-lm
    if not is_vlm and "stop" in options:
        kwargs["stop"] = options["stop"]
    # frequency_penalty / presence_penalty — pass through if present
    for penalty_key in ("frequency_penalty", "presence_penalty"):
        if penalty_key in options and options[penalty_key]:
            kwargs[penalty_key] = options[penalty_key]
    return kwargs


def _inject_tools_into_system(messages: list[dict], tools: list[dict]) -> list[dict]:
    """Inject tool descriptions into the system message when the template doesn't support tools natively."""
    tool_desc_parts = []
    for tool in tools:
        func = tool.get("function", tool)
        name = func.get("name", "unknown")
        desc = func.get("description", "")
        params = func.get("parameters", {})
        tool_desc_parts.append(
            f"- {name}: {desc}\n  Parameters: {json.dumps(params, indent=2)}"
        )
    tool_block = (
        "You have access to the following tools. To call a tool, output a JSON object "
        "with \"name\" and \"arguments\" keys.\n\n"
        "Available tools:\n" + "\n".join(tool_desc_parts)
    )

    messages = list(messages)  # shallow copy
    if messages and messages[0].get("role") == "system":
        messages[0] = {**messages[0], "content": messages[0]["content"] + "\n\n" + tool_block}
    else:
        messages.insert(0, {"role": "system", "content": tool_block})
    return messages


def _apply_chat_template_text(
    tokenizer: Any,
    messages: list[dict],
    tools: list[dict] | None = None,
    caps: TemplateCaps | None = None,
) -> str:
    """Apply chat template for text-only models (mlx-lm).

    Uses TemplateCaps to decide which kwargs to pass, avoiding blind try/except.
    """
    if caps is None:
        caps = TemplateCaps()

    kwargs: dict[str, Any] = {"tokenize": False, "add_generation_prompt": True}

    if tools and caps.supports_tools:
        kwargs["tools"] = tools
        if caps.supports_enable_thinking:
            kwargs["enable_thinking"] = False
    elif tools and not caps.supports_tools:
        # Template doesn't support tools natively — inject into system message
        logger.info("Template lacks tool support, injecting tool descriptions into system message")
        messages = _inject_tools_into_system(messages, tools)

    try:
        return tokenizer.apply_chat_template(messages, **kwargs)
    except Exception as exc:
        # If tools kwarg caused the error, retry without it (injecting instead)
        if tools and "tools" in kwargs:
            logger.warning("apply_chat_template failed with tools kwarg (%s), retrying with injection", exc)
            del kwargs["tools"]
            kwargs.pop("enable_thinking", None)
            messages = _inject_tools_into_system(messages, tools)
            try:
                return tokenizer.apply_chat_template(messages, **kwargs)
            except Exception as exc2:
                raise RuntimeError(
                    f"Chat template failed even without tools: {exc2}"
                ) from exc2
        raise RuntimeError(f"Chat template failed: {exc}") from exc


def _apply_chat_template_vlm(
    processor: Any,
    model: Any,
    messages: list[dict],
    images: list[str] | None = None,
) -> str:
    """Apply chat template for vision-language models (mlx-vlm)."""
    import mlx_vlm

    config = model.config if hasattr(model, "config") else {}
    num_images = len(images) if images else 0
    # Pass the full message list so the model gets proper conversation context
    return mlx_vlm.apply_chat_template(
        processor, config, messages, num_images=num_images
    )


def _extract_images(messages: list[dict]) -> list[str] | None:
    """Extract image URLs/paths from message content."""
    images = []
    for msg in messages:
        if msg.get("images"):
            images.extend(msg["images"])
    return images if images else None


async def generate_completion(
    manager: ModelManager,
    model_name: str,
    prompt: str,
    options: dict | None = None,
    stream: bool = True,
    keep_alive: str | None = None,
    max_tokens: int = 512,
    images: list[str] | None = None,
) -> AsyncGenerator[dict, None] | dict:
    """Generate a text completion, streaming or not."""
    stats = TimingStats()

    with Timer() as load_timer:
        lm = await manager.ensure_loaded(model_name, keep_alive)
    stats.load_duration = load_timer.duration_ns

    gen_kwargs = _build_generate_kwargs(options, is_vlm=lm.is_vlm)
    mt = gen_kwargs.pop("max_tokens", max_tokens)

    if stream:
        return _stream_completion(lm, prompt, mt, gen_kwargs, stats, images)
    else:
        return await _full_completion(lm, prompt, mt, gen_kwargs, stats, images)


async def _stream_completion(
    lm: LoadedModel,
    prompt: str,
    max_tokens: int,
    gen_kwargs: dict,
    stats: TimingStats,
    images: list[str] | None = None,
) -> AsyncGenerator[dict, None]:
    async with _inference_lock:
        mx.clear_cache()
        stream = async_mlx_stream(
            lm.model, lm.tokenizer, prompt,
            max_tokens=max_tokens,
            is_vlm=lm.is_vlm,
            images=images,
            **gen_kwargs,
        )
        try:
            with _inference_ref(lm), Timer() as total_timer:
                with Timer() as eval_timer:
                    async for token in stream:
                        yield {"text": token.text, "done": False}
                        stats.eval_count = token.generation_tokens
                        stats.prompt_eval_count = token.prompt_tokens

                stats.eval_duration = eval_timer.duration_ns

            stats.total_duration = total_timer.duration_ns
            yield {"text": "", "done": True, "stats": stats}
        finally:
            # Shield from cancellation — we MUST wait for the Metal thread
            # to finish before releasing _inference_lock, otherwise the next
            # inference will hit concurrent Metal command buffer access.
            await asyncio.shield(stream.drain_and_join())


async def _full_completion(
    lm: LoadedModel,
    prompt: str,
    max_tokens: int,
    gen_kwargs: dict,
    stats: TimingStats,
    images: list[str] | None = None,
) -> dict:
    async with _inference_lock:
        mx.clear_cache()
        with _inference_ref(lm):
            return await _full_completion_inner(
                lm, prompt, max_tokens, gen_kwargs, stats, images,
            )


async def _full_completion_inner(
    lm: LoadedModel,
    prompt: str,
    max_tokens: int,
    gen_kwargs: dict,
    stats: TimingStats,
    images: list[str] | None = None,
) -> dict:
    with Timer() as total_timer:
        with Timer() as eval_timer:
            if lm.is_vlm:
                import mlx_vlm
                result = await asyncio.to_thread(
                    mlx_vlm.generate,
                    lm.model,
                    lm.tokenizer,
                    prompt=prompt,
                    image=images,
                    max_tokens=max_tokens,
                    **gen_kwargs,
                )
            else:
                import mlx_lm
                result = await asyncio.to_thread(
                    mlx_lm.generate,
                    lm.model,
                    lm.tokenizer,
                    prompt=prompt,
                    max_tokens=max_tokens,
                    **gen_kwargs,
                )

    stats.eval_duration = eval_timer.duration_ns
    stats.total_duration = total_timer.duration_ns

    # mlx_vlm.generate returns GenerationResult dataclass
    if hasattr(result, "text"):
        text = result.text
    elif isinstance(result, str):
        text = result
    else:
        text = str(result)
    return {"text": text, "done": True, "stats": stats}


async def generate_chat(
    manager: ModelManager,
    model_name: str,
    messages: list[dict],
    options: dict | None = None,
    tools: list[dict] | None = None,
    stream: bool = True,
    keep_alive: str | None = None,
    max_tokens: int = 512,
) -> AsyncGenerator[dict, None] | dict:
    """Generate a chat completion."""
    stats = TimingStats()

    with Timer() as load_timer:
        lm = await manager.ensure_loaded(model_name, keep_alive)
    stats.load_duration = load_timer.duration_ns

    images = _extract_images(messages)

    if lm.is_vlm and not tools:
        prompt = _apply_chat_template_vlm(lm.tokenizer, lm.model, messages, images)
    else:
        # Use text template path when tools are needed, even for VLM-loaded models,
        # because _apply_chat_template_vlm doesn't support tool definitions.
        # For VLM-loaded models, try to get the underlying tokenizer which has
        # the proper chat template for tool calling.
        tokenizer = lm.tokenizer
        if lm.is_vlm and hasattr(tokenizer, "tokenizer"):
            tokenizer = tokenizer.tokenizer
        prompt = _apply_chat_template_text(tokenizer, messages, tools, caps=lm.template_caps)
        if tools:
            logger.info("Chat prompt with %d tools", len(tools))
        logger.debug("Prompt (first 1000 chars): %s", prompt[:1000])

    gen_kwargs = _build_generate_kwargs(options, is_vlm=lm.is_vlm)
    mt = gen_kwargs.pop("max_tokens", max_tokens)

    if stream:
        return _stream_completion(lm, prompt, mt, gen_kwargs, stats, images)
    else:
        return await _full_completion(lm, prompt, mt, gen_kwargs, stats, images)


async def generate_embeddings(
    manager: ModelManager,
    model_name: str,
    texts: list[str],
    keep_alive: str | None = None,
) -> list[list[float]]:
    """Generate embeddings using the model's hidden states or embed_tokens layer."""
    import mlx.core as mx

    lm = await manager.ensure_loaded(model_name, keep_alive)

    async with _inference_lock:
        embeddings = []

        tokenizer = lm.tokenizer
        if hasattr(tokenizer, "tokenizer"):
            tokenizer = tokenizer.tokenizer

        # Check if model has a static embedding layer we can use directly
        embed_layer = None
        model_inner = getattr(lm.model, "model", lm.model)
        if hasattr(model_inner, "embed_tokens"):
            embed_layer = model_inner.embed_tokens

        for text in texts:
            tokens = tokenizer.encode(text)
            input_ids = mx.array([tokens])

            if embed_layer is not None:
                # Use static token embeddings — no forward pass needed
                hidden = embed_layer(input_ids)
            else:
                outputs = lm.model(input_ids)
                if hasattr(outputs, "hidden_states") and outputs.hidden_states:
                    hidden = outputs.hidden_states[-1]
                elif hasattr(outputs, "last_hidden_state"):
                    hidden = outputs.last_hidden_state
                else:
                    hidden = outputs

            # Robust shape handling
            if hidden.ndim == 3:
                # (batch, seq, dim) — mean-pool over sequence
                embedding = mx.mean(hidden[0], axis=0)
            elif hidden.ndim == 2:
                # (seq, dim) — mean-pool over sequence
                embedding = mx.mean(hidden, axis=0)
            elif hidden.ndim == 1:
                embedding = hidden
            else:
                raise ValueError(f"Unexpected embedding tensor shape: {hidden.shape}")

            embeddings.append(embedding.tolist())

        return embeddings
