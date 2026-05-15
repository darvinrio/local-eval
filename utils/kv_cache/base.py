from abc import ABC, abstractmethod
from typing import Any
from utils.kv_cache.models import KVCacheEstimate

class KVCacheAdapter(ABC):
    """Base class for model-family-specific KV cache estimators."""

    @staticmethod
    @abstractmethod
    def supported_model_types() -> list[str]:
        """Return list of model_type strings this adapter handles."""
        ...

    @abstractmethod
    def estimate(
        self,
        text_config: dict[str, Any],
        kv_quant_bits: int = 16,
    ) -> KVCacheEstimate:
        """
        Compute a KVCacheEstimate from the resolved text_config.

        Args:
            text_config: The text/language model sub-config dict.
            kv_quant_bits: Bits per KV element. 16 = bf16 (default),
                           8 or 4 for quantised KV cache testing.

        Returns:
            KVCacheEstimate with per-layer-type breakdown.
        """
        ...
