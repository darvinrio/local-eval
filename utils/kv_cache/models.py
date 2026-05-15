"""
utils/kv_cache/models.py

Models for KV cache estimation.
"""

import msgspec


class LayerKVCacheInfo(msgspec.Struct):
    """KV cache cost for one group of same-type layers.

    Layers fall into two categories:
    - **Per-token layers** (full_attention, sliding_attention, gqa):
      Cache grows O(S). Cost = bytes_per_token_per_layer × effective_tokens.
    - **Fixed-state layers** (linear_attention / Gated DeltaNet):
      Maintain a fixed-size state matrix O(1) w.r.t. sequence length.
      Cost = fixed_state_bytes_per_layer (constant, independent of S).
    """

    layer_type: str  # "sliding_attention", "full_attention", "linear_attention", "gqa"
    num_layers: int
    bytes_per_token_per_layer: int  # O(S) cost; 0 for fixed-state layers
    max_cache_tokens: int | None = None  # None = unbounded; else sliding-window cap
    fixed_state_bytes_per_layer: int = (
        0  # O(1) cost for linear attention layers; 0 for standard attention
    )


class KVCacheEstimate(msgspec.Struct):
    """Full-model KV cache estimate with per-layer-type breakdown."""

    model_type: str
    kv_quant_bits: int  # effective bits used (16, 8, 4)
    layer_breakdowns: list[LayerKVCacheInfo]

    @property
    def bytes_per_token(self) -> int:
        """Total bytes/token across O(S) layer types only (uncapped)."""
        return sum(
            lb.num_layers * lb.bytes_per_token_per_layer
            for lb in self.layer_breakdowns
            if lb.bytes_per_token_per_layer > 0
        )

    @property
    def fixed_state_bytes(self) -> int:
        """Total fixed-state bytes across all O(1) layers (context-independent)."""
        return sum(
            lb.num_layers * lb.fixed_state_bytes_per_layer
            for lb in self.layer_breakdowns
            if lb.fixed_state_bytes_per_layer > 0
        )

    def estimate_bytes(self, context_tokens: int) -> int:
        """Total KV cache bytes for a given context length.

        Handles three scaling behaviours:
        - O(S) unbounded:  bytes_per_token_per_layer × S
        - O(S) capped:     bytes_per_token_per_layer × min(S, W)
        - O(1) fixed:      fixed_state_bytes_per_layer  (no S scaling)
        """
        total = 0
        for lb in self.layer_breakdowns:
            # Fixed-state contribution (linear attention)
            total += lb.num_layers * lb.fixed_state_bytes_per_layer
            # Per-token contribution (standard/sliding attention)
            if lb.bytes_per_token_per_layer > 0:
                effective = (
                    min(context_tokens, lb.max_cache_tokens)
                    if lb.max_cache_tokens is not None
                    else context_tokens
                )
                total += lb.num_layers * lb.bytes_per_token_per_layer * effective

        if total == 0:
            raise ValueError(
                "KV cache estimate is zero. Please check layer_breakdowns."
            )
        return total

    def estimate_gb(self, context_tokens: int) -> float:
        """Convenience: estimate in GiB."""
        return self.estimate_bytes(context_tokens) / (1024**3)
