import asyncio
import logging
import threading
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class StreamToken:
    text: str
    token: int | None
    prompt_tokens: int
    generation_tokens: int
    prompt_tps: float
    generation_tps: float
    finish_reason: str | None = None


_SENTINEL = object()
_ERROR_KEY = "__error__"


async def async_mlx_stream(
    model: Any,
    tokenizer: Any,
    prompt: str,
    max_tokens: int = 512,
    is_vlm: bool = False,
    images: list[str] | None = None,
    **kwargs: Any,
) -> AsyncGenerator[StreamToken, None]:
    """Bridge sync mlx_lm/mlx_vlm stream_generate into an async generator."""
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue(maxsize=32)

    def _run():
        try:
            if is_vlm:
                import mlx_vlm
                gen = mlx_vlm.stream_generate(
                    model,
                    tokenizer,
                    prompt=prompt,
                    image=images,
                    max_tokens=max_tokens,
                    **kwargs,
                )
                for resp in gen:
                    tok = StreamToken(
                        text=resp.text,
                        token=getattr(resp, "token", None),
                        prompt_tokens=resp.prompt_tokens,
                        generation_tokens=resp.generation_tokens,
                        prompt_tps=resp.prompt_tps,
                        generation_tps=resp.generation_tps,
                        finish_reason=None,
                    )
                    asyncio.run_coroutine_threadsafe(queue.put(tok), loop).result()
            else:
                import mlx_lm
                for resp in mlx_lm.stream_generate(
                    model,
                    tokenizer,
                    prompt=prompt,
                    max_tokens=max_tokens,
                    **kwargs,
                ):
                    tok = StreamToken(
                        text=resp.text,
                        token=resp.token,
                        prompt_tokens=resp.prompt_tokens,
                        generation_tokens=resp.generation_tokens,
                        prompt_tps=resp.prompt_tps,
                        generation_tps=resp.generation_tps,
                        finish_reason=resp.finish_reason,
                    )
                    asyncio.run_coroutine_threadsafe(queue.put(tok), loop).result()
        except Exception as exc:
            import traceback
            asyncio.run_coroutine_threadsafe(
                queue.put({
                    _ERROR_KEY: str(exc),
                    "__exc_type__": type(exc).__name__,
                    "__traceback__": traceback.format_exc(),
                }), loop
            ).result()
        finally:
            asyncio.run_coroutine_threadsafe(queue.put(_SENTINEL), loop).result()

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()

    while True:
        item = await queue.get()
        if item is _SENTINEL:
            break
        if isinstance(item, dict) and _ERROR_KEY in item:
            exc_type = item.get("__exc_type__", "RuntimeError")
            tb = item.get("__traceback__", "")
            if tb:
                logger.error("Inference error (%s): %s\n%s", exc_type, item[_ERROR_KEY], tb)
            raise RuntimeError(f"{exc_type}: {item[_ERROR_KEY]}")
        yield item
