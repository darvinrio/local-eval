"""
utils/kv_cache/__init__.py

Convenience module for KV cache estimation.
"""

from typing import Any

from utils.kv_cache.adapters import get_adapter
from utils.kv_cache.models import KVCacheEstimate

__all__ = [
    "estimate_kv_cache",
    "KVCacheEstimate",
]


def estimate_kv_cache(
    text_config: dict[str, Any],
    kv_quant_bits: int = 16,
) -> KVCacheEstimate:
    """
    One-call convenience: resolve adapter from config's model_type and estimate.

    Args:
        text_config: The text/language model sub-config dict.
        kv_quant_bits: Bits per KV element. 16 = bf16 (default),
                       8 or 4 for quantised KV cache testing.

    Returns:
        KVCacheEstimate with per-layer-type breakdown.
    """
    model_type = text_config.get("model_type", "")
    adapter = get_adapter(model_type)
    return adapter.estimate(text_config, kv_quant_bits=kv_quant_bits)
