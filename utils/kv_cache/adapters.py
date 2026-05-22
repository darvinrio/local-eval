"""
utils/kv_cache/adapters.py

Concrete KVCacheAdapter implementations per model family.
"""

from typing import Any

from loguru import logger

from utils.kv_cache.base import KVCacheAdapter
from utils.kv_cache.models import KVCacheEstimate, LayerKVCacheInfo

_ADAPTER_REGISTRY: dict[str, type[KVCacheAdapter]] = {}


def register_adapter(cls: type[KVCacheAdapter]) -> type[KVCacheAdapter]:
    """Decorator: register an adapter for its supported model_types."""
    for mt in cls.supported_model_types():
        _ADAPTER_REGISTRY[mt] = cls
    return cls


class StandardGQAAdapter(KVCacheAdapter):
    """
    Fallback adapter for any model type not explicitly handled.

    Assumes standard multi-head attention with Grouped Query Attention (GQA)
    where the number of key/value heads is smaller than the number of attention heads.
    """

    @staticmethod
    def supported_model_types() -> list[str]:
        """Return the list of supported model types."""
        return ["__fallback__"]

    def estimate(
        self,
        text_config: dict[str, Any],
        kv_quant_bits: int = 16,
    ) -> KVCacheEstimate:
        """Estimate the KV cache for a given model configuration."""
        num_layers = text_config.get(
            "num_hidden_layers", text_config.get("num_layers", 0)
        )
        num_kv_heads = text_config.get(
            "num_key_value_heads", text_config.get("num_attention_heads", 0)
        )
        head_dim = text_config.get("head_dim", 0)

        if (
            head_dim == 0
            and "hidden_size" in text_config
            and "num_attention_heads" in text_config
        ):
            head_dim = text_config["hidden_size"] // text_config["num_attention_heads"]

        dtype_bytes = kv_quant_bits / 8

        bpt = int(2 * num_kv_heads * head_dim * dtype_bytes)

        if bpt == 0:
            logger.warning(
                "KV cache estimate is 0. Config may be missing expected keys."
            )
            bpt = 1  # Rough fallback to avoid div by zero issues if any

        info = LayerKVCacheInfo(
            layer_type="gqa",
            num_layers=num_layers,
            bytes_per_token_per_layer=bpt,
            max_cache_tokens=None,
            fixed_state_bytes_per_layer=0,
        )

        return KVCacheEstimate(
            model_type=text_config.get("model_type", "__fallback__"),
            kv_quant_bits=kv_quant_bits,
            layer_breakdowns=[info],
        )


@register_adapter
class Gemma4Adapter(KVCacheAdapter):
    """
    Adapter for Gemma 4 and Gemma 4-based models.

    Gemma 4 uses a hybrid KV cache strategy with:
    - Full attention for the first `num_full` layers
    - Sliding window attention for the remaining `num_sliding` layers
    """

    @staticmethod
    def supported_model_types() -> list[str]:
        """Return the list of supported model types."""
        return ["gemma4", "gemma4_text"]

    def estimate(
        self,
        text_config: dict[str, Any],
        kv_quant_bits: int = 16,
    ) -> KVCacheEstimate:
        """Estimate the KV cache for a given model configuration.

        Gemma 4 has two distinct layer types with different cache geometries:

        - **SWA layers**: Standard K+V caching (2×) using ``num_key_value_heads``
          and ``head_dim``.  Cache capped at ``sliding_window`` tokens.
        - **Global/full layers**: Standard K+V caching (2×) using
          ``num_global_key_value_heads`` and ``global_head_dim``.
          Cache grows unbounded.

        .. note::

            ``attention_k_eq_v=True`` in the HF config only means K and V
            projections share the same shape/dimensions — it does **not**
            confirm that the runtime stores them as a single unified tensor.
            We default to the conservative 2× estimate (separate K and V caches)
            to match standard Transformer KV cache semantics.

            Unified KV (1×) is backend-dependent and unconfirmed for MLX/llama.cpp.
            See ``docs/kv_calc/models/gemma4_26a4b_kv_calc.md`` for the full analysis.
        """
        layer_types = text_config.get("layer_types", [])
        num_sliding = layer_types.count("sliding_attention")
        num_full = layer_types.count("full_attention")

        dtype_bytes = kv_quant_bits / 8
        sliding_window = text_config.get("sliding_window", None)

        # --- SWA layers: standard K + V (2×) ---
        num_kv_heads = text_config.get("num_key_value_heads", 0)
        head_dim = text_config.get("head_dim", 0)
        swa_bpt = int(2 * num_kv_heads * head_dim * dtype_bytes)

        # --- Full/global layers: standard K + V (2×), using main KV heads ---
        # The global KV heads (num_global_key_value_heads=2, global_head_dim=512)
        # are a compressed/auxiliary representation. Full attention layers cache
        # using the standard num_key_value_heads and head_dim, same as SWA layers.
        # See docs/kv_calc/models/gemma4_26a4b_kv_calc.md for analysis.
        full_bpt = int(2 * num_kv_heads * head_dim * dtype_bytes)

        breakdowns = []
        if num_full > 0:
            breakdowns.append(
                LayerKVCacheInfo(
                    layer_type="full_attention",
                    num_layers=num_full,
                    bytes_per_token_per_layer=full_bpt,
                    max_cache_tokens=None,
                    fixed_state_bytes_per_layer=0,
                )
            )

        if num_sliding > 0:
            breakdowns.append(
                LayerKVCacheInfo(
                    layer_type="sliding_attention",
                    num_layers=num_sliding,
                    bytes_per_token_per_layer=swa_bpt,
                    max_cache_tokens=sliding_window,
                    fixed_state_bytes_per_layer=0,
                )
            )

        return KVCacheEstimate(
            model_type=text_config.get("model_type", "gemma4"),
            kv_quant_bits=kv_quant_bits,
            layer_breakdowns=breakdowns,
        )


@register_adapter
class Qwen36HybridAdapter(KVCacheAdapter):
    """
    Adapter for Qwen 3.5 and related hybrid models.

    These models use a hybrid KV cache strategy with:
    - Full attention for standard transformer layers
    - Gated DeltaNet (linear attention) for efficient long-context processing
    """

    @staticmethod
    def supported_model_types() -> list[str]:
        """Return the list of supported model types."""
        return ["qwen3_5", "qwen3_5_text", "qwen3_5_moe", "qwen3_5_moe_text"]

    def estimate(
        self,
        text_config: dict[str, Any],
        kv_quant_bits: int = 16,
    ) -> KVCacheEstimate:
        """Estimate the KV cache for a given model configuration."""
        layer_types = text_config.get("layer_types", [])
        num_full = layer_types.count("full_attention")
        num_linear = layer_types.count("linear_attention")

        dtype_bytes = kv_quant_bits / 8
        breakdowns = []

        if num_full > 0:
            num_kv_heads = text_config.get("num_key_value_heads", 0)
            head_dim = text_config.get("head_dim", 0)
            bpt = int(2 * num_kv_heads * head_dim * dtype_bytes)
            breakdowns.append(
                LayerKVCacheInfo(
                    layer_type="full_attention",
                    num_layers=num_full,
                    bytes_per_token_per_layer=bpt,
                    max_cache_tokens=None,
                    fixed_state_bytes_per_layer=0,
                )
            )

        if num_linear > 0:
            linear_num_key_heads = text_config.get("linear_num_key_heads", 0)
            linear_key_head_dim = text_config.get("linear_key_head_dim", 0)
            linear_num_value_heads = text_config.get("linear_num_value_heads", 0)
            linear_value_head_dim = text_config.get("linear_value_head_dim", 0)

            fixed_state_bytes = int(
                (
                    linear_num_key_heads * linear_key_head_dim
                    + linear_num_value_heads * linear_value_head_dim
                )
                * dtype_bytes
            )

            breakdowns.append(
                LayerKVCacheInfo(
                    layer_type="linear_attention",
                    num_layers=num_linear,
                    bytes_per_token_per_layer=0,
                    max_cache_tokens=None,
                    fixed_state_bytes_per_layer=fixed_state_bytes,
                )
            )

        return KVCacheEstimate(
            model_type=text_config.get("model_type", "qwen3_5"),
            kv_quant_bits=kv_quant_bits,
            layer_breakdowns=breakdowns,
        )


def get_adapter(model_type: str) -> KVCacheAdapter:
    """
    Look up and instantiate the right adapter for a model_type.

    Falls back to StandardGQAAdapter if no specific match.
    """
    cls = _ADAPTER_REGISTRY.get(model_type, StandardGQAAdapter)
    if cls is StandardGQAAdapter and model_type:
        logger.warning(
            f"No specific adapter for {model_type}, falling back to StandardGQAAdapter."
        )
    return cls()
