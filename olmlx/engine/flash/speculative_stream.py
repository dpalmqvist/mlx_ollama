"""Streaming adapter for speculative decoding.

Bridges SpeculativeFlashDecoder into the CancellableStream / StreamToken
contract used by the inference pipeline.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Generator
from typing import Any

import mlx.core as mx

from olmlx.engine.flash.speculative import SpeculativeFlashDecoder
from olmlx.utils.streaming import StreamToken


def _decode_incremental(
    tokenizer: Any, generated: list[int], prev_text: str
) -> tuple[str, str]:
    """Decode new text incrementally by diffing against previous decode."""
    if tokenizer is None:
        return "", prev_text
    full_text = (
        tokenizer.decode(generated[-16:])
        if len(generated) > 16
        else tokenizer.decode(generated)
    )
    if len(generated) > 16:
        # For long sequences, decode a small window and diff
        prefix = tokenizer.decode(generated[-17:-1]) if len(generated) > 17 else ""
        new_text = full_text[len(prefix) :]
    else:
        new_text = full_text[len(prev_text) :]
        prev_text = full_text
    return new_text, prev_text


def speculative_stream_generate(
    decoder: SpeculativeFlashDecoder,
    prompt_tokens: list[int],
    max_tokens: int,
    cancel_event: threading.Event,
    eos_token_id: int | None = None,
    tokenizer: Any = None,
) -> Generator[StreamToken, None, None]:
    """Sync generator that yields StreamToken objects for speculative decoding.

    Args:
        decoder: SpeculativeFlashDecoder with prefill/step API.
        prompt_tokens: Token IDs for the prompt.
        max_tokens: Maximum number of tokens to generate.
        cancel_event: Set to stop generation.
        eos_token_id: Stop generation when this token is produced.
        tokenizer: Tokenizer for incremental text decoding. If None, text is empty.
    """
    prompt_arr = mx.array([prompt_tokens])
    prompt_len = len(prompt_tokens)

    t0 = time.perf_counter()
    first_token = decoder.prefill(prompt_arr)

    generated: list[int] = [first_token]
    gen_count = 1
    elapsed = time.perf_counter() - t0

    new_text, prev_text = _decode_incremental(tokenizer, generated, "")

    yield StreamToken(
        text=new_text,
        token=first_token,
        prompt_tokens=prompt_len,
        generation_tokens=gen_count,
        prompt_tps=prompt_len / max(elapsed, 1e-9),
        generation_tps=gen_count / max(elapsed, 1e-9),
    )

    if (
        eos_token_id is not None and first_token == eos_token_id
    ) or cancel_event.is_set():
        return

    while gen_count < max_tokens:
        if cancel_event.is_set():
            break

        accepted, _ = decoder.step()

        for token in accepted:
            if gen_count >= max_tokens:
                break

            gen_count += 1
            generated.append(token)
            elapsed = time.perf_counter() - t0

            new_text, prev_text = _decode_incremental(tokenizer, generated, prev_text)

            finish = None
            if gen_count >= max_tokens:
                finish = "length"
            elif eos_token_id is not None and token == eos_token_id:
                finish = "stop"

            yield StreamToken(
                text=new_text,
                token=token,
                prompt_tokens=prompt_len,
                generation_tokens=gen_count,
                prompt_tps=prompt_len / max(elapsed, 1e-9),
                generation_tps=gen_count / max(elapsed, 1e-9),
                finish_reason=finish,
            )

            if finish == "stop":
                return


def async_speculative_stream(
    decoder: SpeculativeFlashDecoder,
    tokenizer: Any,
    prompt: str | list[int],
    max_tokens: int,
) -> Any:
    """Create a CancellableStream for speculative decoding.

    Matches the interface of async_mlx_stream from utils/streaming.py.
    """
    from olmlx.utils.streaming import CancellableStream

    if isinstance(prompt, str):
        prompt_tokens = tokenizer.encode(prompt)
    else:
        prompt_tokens = prompt

    eos_token_id = getattr(tokenizer, "eos_token_id", None)

    def gen_factory(cancel_event: threading.Event):
        return speculative_stream_generate(
            decoder,
            prompt_tokens,
            max_tokens=max_tokens,
            cancel_event=cancel_event,
            eos_token_id=eos_token_id,
            tokenizer=tokenizer,
        )

    stream = CancellableStream(gen_factory)
    stream.start()
    return stream
