from typing import Any

from utils.kv_cache.models import KVCacheEstimate, LayerKVCacheInfo
from utils.kv_cache.base import KVCacheAdapter
from utils.kv_cache.adapters import get_adapter

def estimate_kv_cache(
    text_config: dict[str, Any],
    kv_quant_bits: int = 16,
) -> KVCacheEstimate:
    """
    One-call convenience: resolve adapter from config's model_type and estimate.
    """
    model_type = text_config.get("model_type", "")
    adapter = get_adapter(model_type)
    return adapter.estimate(text_config, kv_quant_bits=kv_quant_bits)
