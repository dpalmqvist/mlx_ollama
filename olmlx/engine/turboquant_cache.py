"""TurboQuant KV cache: drop-in replacement for mlx-lm's KVCache.

Stores quantized key/value vectors using TurboQuant_mse and dequantizes
on fetch, providing transparent memory compression.
"""

from __future__ import annotations

import logging
from typing import Any

import mlx.core as mx

from mlx_lm.models.cache import _BaseCache, create_attention_mask

from olmlx.engine.turboquant import (
    TurboQuantRotation,
    turboquant_dequantize,
    turboquant_quantize,
)

logger = logging.getLogger(__name__)


class TurboQuantKVCache(_BaseCache):
    """KV cache with TurboQuant compression.

    Quantizes K/V vectors on store and dequantizes on fetch.
    Implements the same interface as mlx-lm's KVCache.
    """

    step = 256

    def __init__(
        self,
        bits: int,
        rotation_key: TurboQuantRotation,
        rotation_value: TurboQuantRotation,
    ):
        self.bits = bits
        self.rotation_key = rotation_key
        self.rotation_value = rotation_value
        # Quantized storage (compact)
        self._key_indices: mx.array | None = None
        self._key_norms: mx.array | None = None
        self._value_indices: mx.array | None = None
        self._value_norms: mx.array | None = None
        # Dequantized cache (avoids O(n^2) re-dequantization)
        self._keys_deq: mx.array | None = None
        self._values_deq: mx.array | None = None
        self.offset = 0

    def update_and_fetch(
        self, keys: mx.array, values: mx.array
    ) -> tuple[mx.array, mx.array]:
        """Quantize new K/V, append to store, return dequantized cache.

        Only the new tokens are dequantized each call — the previously
        dequantized prefix is reused, making per-step cost O(new_tokens)
        instead of O(total_tokens).
        """
        B, n_heads, num_steps, head_dim = keys.shape
        prev = self.offset

        # Quantize incoming tokens
        k_idx, k_nrm = turboquant_quantize(keys, self.rotation_key, self.bits)
        v_idx, v_nrm = turboquant_quantize(values, self.rotation_value, self.bits)

        # Dequantize only the new tokens
        k_new_deq = turboquant_dequantize(k_idx, k_nrm, self.rotation_key, self.bits)
        v_new_deq = turboquant_dequantize(v_idx, v_nrm, self.rotation_value, self.bits)

        # Allocate or expand quantized buffers
        if self._key_indices is None or (prev + num_steps) > self._key_indices.shape[2]:
            new_steps = (self.step + num_steps - 1) // self.step * self.step
            idx_shape = (B, n_heads, new_steps, head_dim)
            nrm_shape = (B, n_heads, new_steps, 1)
            deq_shape = (B, n_heads, new_steps, head_dim)

            if self._key_indices is not None:
                if prev % self.step != 0:
                    self._key_indices = self._key_indices[..., :prev, :]
                    self._key_norms = self._key_norms[..., :prev, :]
                    self._value_indices = self._value_indices[..., :prev, :]
                    self._value_norms = self._value_norms[..., :prev, :]
                    self._keys_deq = self._keys_deq[..., :prev, :]
                    self._values_deq = self._values_deq[..., :prev, :]
                self._key_indices = mx.concatenate(
                    [self._key_indices, mx.zeros(idx_shape, dtype=mx.uint8)], axis=2
                )
                self._key_norms = mx.concatenate(
                    [self._key_norms, mx.zeros(nrm_shape, dtype=mx.float16)], axis=2
                )
                self._value_indices = mx.concatenate(
                    [self._value_indices, mx.zeros(idx_shape, dtype=mx.uint8)], axis=2
                )
                self._value_norms = mx.concatenate(
                    [self._value_norms, mx.zeros(nrm_shape, dtype=mx.float16)], axis=2
                )
                self._keys_deq = mx.concatenate(
                    [self._keys_deq, mx.zeros(deq_shape, dtype=keys.dtype)], axis=2
                )
                self._values_deq = mx.concatenate(
                    [self._values_deq, mx.zeros(deq_shape, dtype=values.dtype)], axis=2
                )
            else:
                self._key_indices = mx.zeros(idx_shape, dtype=mx.uint8)
                self._key_norms = mx.zeros(nrm_shape, dtype=mx.float16)
                self._value_indices = mx.zeros(idx_shape, dtype=mx.uint8)
                self._value_norms = mx.zeros(nrm_shape, dtype=mx.float16)
                self._keys_deq = mx.zeros(deq_shape, dtype=keys.dtype)
                self._values_deq = mx.zeros(deq_shape, dtype=values.dtype)

        # Store quantized data and dequantized cache
        self.offset += num_steps
        self._key_indices[..., prev : self.offset, :] = k_idx
        self._key_norms[..., prev : self.offset, :] = k_nrm
        self._value_indices[..., prev : self.offset, :] = v_idx
        self._value_norms[..., prev : self.offset, :] = v_nrm
        self._keys_deq[..., prev : self.offset, :] = k_new_deq
        self._values_deq[..., prev : self.offset, :] = v_new_deq

        return (
            self._keys_deq[..., : self.offset, :],
            self._values_deq[..., : self.offset, :],
        )

    def is_trimmable(self):
        return True

    def trim(self, n: int) -> int:
        n = min(self.offset, n)
        self.offset -= n
        return n

    def make_mask(self, *args, **kwargs):
        return create_attention_mask(*args, offset=self.offset, **kwargs)

    def empty(self):
        return self._key_indices is None


def _detect_head_dim(model: Any) -> int:
    """Detect head_dim from model args or K projection layer shape.

    Handles models where head_dim != hidden_size // num_attention_heads
    (e.g. Gemma 3, Phi-3/4).
    """
    # Prefer explicit head_dim from model args
    head_dim = getattr(model.args, "head_dim", None)
    if head_dim is not None:
        return head_dim

    # Derive from K projection weight shape: k_proj.weight is (n_kv_heads * head_dim, hidden_size)
    try:
        layer = model.layers[0]
        k_proj = layer.self_attn.k_proj
        weight = k_proj.weight
        if isinstance(weight, mx.array):
            kv_out_dim = weight.shape[0]
            n_kv_heads = getattr(model.args, "num_key_value_heads", None)
            if n_kv_heads:
                return kv_out_dim // n_kv_heads
    except (AttributeError, IndexError):
        pass

    # Last resort: hidden_size // num_attention_heads
    return model.args.hidden_size // model.args.num_attention_heads


def make_turboquant_cache(model: Any, bits: int) -> list[TurboQuantKVCache]:
    """Create a list of TurboQuantKVCache objects, one per model layer."""
    num_layers = len(model.layers)
    head_dim = _detect_head_dim(model)

    caches = []
    for i in range(num_layers):
        rot_k = TurboQuantRotation(head_dim=head_dim, seed=i * 2)
        rot_v = TurboQuantRotation(head_dim=head_dim, seed=i * 2 + 1)
        caches.append(TurboQuantKVCache(bits=bits, rotation_key=rot_k, rotation_value=rot_v))

    logger.info(
        "Created TurboQuant KV cache: %d layers, %d-bit, head_dim=%d",
        num_layers,
        bits,
        head_dim,
    )
    return caches
