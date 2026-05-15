# 004 — KV Cache Adapter Refactor

**Status**: Draft
**Created**: 2026-05-13
**Last Updated**: 2026-05-14
**Related**: `specs/003-oom-resilience-and-preflight-fix.md`, `docs/kv-cache-calc-research.md`

---

## 1. Problem

The current `_memory_preflight` in `benchmarks/mlx_ctx_bm.py` (L90-180) has two issues:

1. **Naive KV cache formula** — treats all layers identically using a single `max()` over heterogeneous config keys. This produces wildly inaccurate estimates for hybrid architectures (Gemma4 with sliding-window + unified K=V global layers; Qwen3.6 with linear_attention + full_attention layers).
2. **Tightly coupled** — KV estimation, config resolution, and memory-budget decisions are all fused into one private function inside the benchmark runner.

## 2. Goals

| # | Goal |
|---|------|
| G1 | Per-layer-type KV cache breakdown (bytes/token by `layer_type`) |
| G2 | Sliding-window capping for tighter estimates |
| G3 | Configurable `kv_quant_bits` (default from config, override for 4/8-bit KV testing) |
| G4 | Extensible adapter pattern — new model families added without touching the benchmark |
| G5 | Extract preflight: adapters own estimation, benchmark owns the go/no-go decision |

## 3. Architecture

```
utils/
  basic.py              # existing
  kv_cache/
    __init__.py          # public API re-exports
    models.py            # LayerKVCacheInfo, KVCacheEstimate
    base.py              # KVCacheAdapter (ABC)
    adapters.py          # Gemma4, Qwen36Hybrid, StandardGQA + registry
```

### 3.1 Data Models (`models.py`)

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class LayerKVCacheInfo:
    """KV cache cost for one group of same-type layers.

    Layers fall into two categories:
    - **Per-token layers** (full_attention, sliding_attention, gqa):
      Cache grows O(S). Cost = bytes_per_token_per_layer × effective_tokens.
    - **Fixed-state layers** (linear_attention / Gated DeltaNet):
      Maintain a fixed-size state matrix O(1) w.r.t. sequence length.
      Cost = fixed_state_bytes_per_layer (constant, independent of S).
    """
    layer_type: str                        # "sliding_attention", "full_attention", "linear_attention", "gqa"
    num_layers: int
    bytes_per_token_per_layer: int         # O(S) cost; 0 for fixed-state layers
    max_cache_tokens: int | None = None    # None = unbounded; else sliding-window cap
    fixed_state_bytes_per_layer: int = 0   # O(1) cost for linear attention layers; 0 for standard attention

@dataclass
class KVCacheEstimate:
    """Full-model KV cache estimate with per-layer-type breakdown."""
    model_type: str
    kv_quant_bits: int               # effective bits used (16, 8, 4)
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
        return total

    def estimate_gb(self, context_tokens: int) -> float:
        """Convenience: estimate in GiB."""
        return self.estimate_bytes(context_tokens) / (1024 ** 3)
```

### 3.2 Abstract Adapter (`base.py`)

```python
from abc import ABC, abstractmethod
from typing import Any

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
```

> **Note:** The adapter receives the **already-resolved** `text_config` (i.e., `_resolve_text_config` runs first). The adapter does not deal with multimodal wrapper nesting.

### 3.3 Concrete Adapters (`adapters.py`)

#### 3.3.1 `Gemma4Adapter`

**Supported model_types**: `"gemma4"`, `"gemma4_text"`

Reads from config:
- `layer_types` list → count `sliding_attention` vs `full_attention`
- `num_key_value_heads`, `head_dim` → KV dims for **both** SWA and full layers
- `sliding_window` → cap for sliding layers

> **Note on `attention_k_eq_v` and global heads:** The `num_global_key_value_heads` and `global_head_dim` config keys describe the **query-side** geometry of global layers, not the KV-cache storage. Both SWA and full-attention layers store K and V using `num_key_value_heads × head_dim`. See `docs/kv-cache-calc-research.md` §3.

**Formulas** (per the research doc):

| Layer type | Formula (bytes/layer/token) |
|---|---|
| `sliding_attention` | `2 × num_kv_heads × head_dim × dtype_bytes` |
| `full_attention` | `2 × num_kv_heads × head_dim × dtype_bytes` |

Where `dtype_bytes = kv_quant_bits / 8`.

Sliding layers get `max_cache_tokens = sliding_window`. Full-attention layers are unbounded (`None`).

**Validation against research doc** (Gemma4 26B-A4B, bf16, `num_kv_heads=8, head_dim=256`):
```
bpt_layer = 2 × 8 × 256 × 2 = 8,192 bytes/token/layer  (same for both types)

Full layers:  5 × 8,192 × S = 40,960 × S   (unbounded)
SWA layers:  25 × 8,192 × min(S, 1024) → capped at 25 × 8,192 × 1024 = 209,715,200 bytes

@ 128K context:
  Full:  5 × 8,192 × 131,072 = 5,368,709,120 bytes  (5.0 GiB)
  SWA:  25 × 8,192 × 1,024   =   209,715,200 bytes  (0.20 GiB)  ← capped
  Total = 5,578,424,320 bytes ≈ 5.20 GiB
```

> **Note:** The research doc's Gemma4-26B tables show `81,920 bytes/token` for full layers (using H_kv=16 from the 31B variant). With the actual 26B config (`H_kv=8`), the per-token rate halves. The adapter reads `num_key_value_heads` directly from config, so it auto-corrects for this.

#### 3.3.2 `Qwen36HybridAdapter`

**Supported model_types**: `"qwen3_5"`, `"qwen3_5_text"`, `"qwen3_5_moe"`, `"qwen3_5_moe_text"`

Handles **both** dense (Qwen 3.6-27B) and MoE (Qwen 3.6-35B-A3B) variants — they share the same hybrid Gated DeltaNet + Gated Attention architecture.

Reads from config:
- `layer_types` list → count `full_attention` vs `linear_attention`
- `num_key_value_heads`, `head_dim` → for full_attention layers (O(S) KV cache)
- `linear_num_key_heads`, `linear_key_head_dim` → K dims for linear layers (O(1) state)
- `linear_num_value_heads`, `linear_value_head_dim` → V dims for linear layers (O(1) state)

**Formulas**:

| Layer type | Scaling | Formula |
|---|---|---|
| `full_attention` | O(S) per-token | `bytes_per_token_per_layer = 2 × num_kv_heads × head_dim × dtype_bytes` |
| `linear_attention` | **O(1) fixed-state** | `fixed_state_bytes_per_layer = (linear_num_key_heads × linear_key_head_dim + linear_num_value_heads × linear_value_head_dim) × dtype_bytes` |

> **Critical:** Linear attention (Gated DeltaNet) layers maintain a **fixed-size state matrix** that does NOT grow with sequence length. They are O(1), not O(S). The state matrix is updated incrementally for each new token. `bytes_per_token_per_layer = 0` for these layers. See `docs/kv-cache-calc-research.md` §1 and §3.

**Validation** (Qwen3.6 35B-A3B, bf16):
```
full_attn (O(S)):     10 layers × (2 × 2 × 256 × 2)         = 10 × 2,048  = 20,480 bytes/token
linear_attn (O(1)):   30 layers × ((16×128 + 32×128) × 2)    = 30 × 12,288 = 368,640 bytes FIXED

@ 128K (131,072 tokens):
  Full-attn KV:     20,480 × 131,072 = 2,684,354,560 bytes  (2.50 GiB)
  Linear state:     368,640 bytes                            (0.00035 GiB)
  Total ≈ 2.50 GiB                                          ✓ matches research doc
```

**Validation** (Qwen3.6 27B Dense, bf16):
```
full_attn (O(S)):     16 layers × (2 × 4 × 256 × 2)         = 16 × 4,096  = 65,536 bytes/token
linear_attn (O(1)):   48 layers × ((16×128 + 48×128) × 2)    = 48 × 16,384 = 786,432 bytes FIXED

@ 128K: 65,536 × 131,072 + 786,432 ≈ 8.01 GB ≈ 7.46 GiB    ✓ matches research doc
```

#### 3.3.3 `StandardGQAAdapter` (Fallback)

**Supported model_types**: `"__fallback__"` (used when no specific adapter matches)

Reads:
- `num_hidden_layers` / `num_layers`
- `num_key_value_heads` (falls back to `num_attention_heads` for MHA)
- `head_dim` (falls back to `hidden_size // num_attention_heads`)

**Formula**: `2 x num_kv_heads x head_dim x dtype_bytes` per layer, all unbounded.

This covers Llama, Qwen3 (non-hybrid), Mistral, etc.

### 3.4 Registry

```python
# In adapters.py

_ADAPTER_REGISTRY: dict[str, type[KVCacheAdapter]] = {}

def register_adapter(cls: type[KVCacheAdapter]) -> type[KVCacheAdapter]:
    """Decorator: register an adapter for its supported model_types."""
    for mt in cls.supported_model_types():
        _ADAPTER_REGISTRY[mt] = cls
    return cls

def get_adapter(model_type: str) -> KVCacheAdapter:
    """
    Look up and instantiate the right adapter for a model_type.

    Falls back to StandardGQAAdapter if no specific match.
    """
    cls = _ADAPTER_REGISTRY.get(model_type, StandardGQAAdapter)
    return cls()
```

### 3.5 Public API (`__init__.py`)

```python
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
```

## 4. Benchmark Integration

### 4.1 Changes to `mlx_ctx_bm.py`

**Delete**: The KV cache math inside `_memory_preflight` (lines ~114-151).

**Replace with**:
```python
from utils.kv_cache import estimate_kv_cache

def _memory_preflight(
    task: ContextTask,
    context_tokens: int,
    text_config: dict[str, Any],
    force_run: bool,
    safety_threshold: float,
    generation_headroom_gb: float,
    pre_load_available_gb: float,
    kv_quant_bits: int = 16,
) -> tuple[bool, str]:
    """Go/no-go memory decision. KV estimation delegated to adapter."""
    estimate = estimate_kv_cache(text_config, kv_quant_bits=kv_quant_bits)
    kv_cache_gb = estimate.estimate_gb(context_tokens)

    model_weights_gb = mx.get_active_memory() / (1024**3)
    required_gb = model_weights_gb + kv_cache_gb + generation_headroom_gb

    logger.debug(
        f"Preflight: {estimate.model_type} kv_bits={estimate.kv_quant_bits} "
        f"kv_cache={kv_cache_gb:.2f} GiB | breakdown: "
        + ", ".join(
            f"{lb.layer_type}({lb.num_layers}L)={lb.bytes_per_token_per_layer}B/tok/L"
            + (f" cap={lb.max_cache_tokens}" if lb.max_cache_tokens else "")
            for lb in estimate.layer_breakdowns
        )
    )

    # ... existing threshold comparison + force_run prompt unchanged ...
```

### 4.2 Config Extension (`MLXContextConfig`)

Add optional field:
```python
kv_quant_bits: int = 16  # bf16 default; set to 8 or 4 for KV quant testing
```

This is threaded from `MLXContextConfig` -> `_memory_preflight` -> `estimate_kv_cache`.

## 5. File Plan

| File | Action | Description |
|---|---|---|
| `utils/kv_cache/__init__.py` | Create | Public API: `estimate_kv_cache`, re-exports |
| `utils/kv_cache/models.py` | Create | `LayerKVCacheInfo`, `KVCacheEstimate` dataclasses |
| `utils/kv_cache/base.py` | Create | `KVCacheAdapter` ABC |
| `utils/kv_cache/adapters.py` | Create | `Gemma4Adapter`, `Qwen36HybridAdapter`, `StandardGQAAdapter`, registry |
| `benchmarks/mlx_ctx_bm.py` | Modify | Replace KV math in `_memory_preflight` with adapter call; add `kv_quant_bits` param |
| `models/config.py` | Modify | Add `kv_quant_bits: int = 16` to `MLXContextConfig` |

## 6. Testing Strategy

- **Unit tests** for each adapter using the sample configs in `output/samples/`.
- Validate against the reference numbers in `docs/kv-cache-calc-research.md`:
  - Gemma4 26B-A4B @ 128K: ≈5.20 GiB (bf16)
  - Qwen3.6 35B-A3B @ 128K: ≈2.50 GiB (bf16, O(S) component only)
  - Qwen3.6 27B @ 128K: ≈7.46 GiB (bf16)
  - Qwen3 8B (standard GQA fallback): 144 KiB/token
- Test `kv_quant_bits` override: e.g., Gemma4 at 8-bit should yield exactly half of bf16.
- Test sliding-window capping: at `context_tokens < sliding_window`, result matches uncapped; at `context_tokens > sliding_window`, sliding layers are capped.
- Test linear attention fixed-state: verify `estimate_bytes(4096) - estimate_bytes(0)` equals only the full-attention O(S) contribution (linear layers must NOT scale with S).

## 7. Open Decisions

| # | Question | Default |
|---|---|---|
| O1 | Should `_resolve_text_config` also move to `utils/kv_cache/`? | **No** — keep in benchmark; adapters receive already-resolved config |
| O2 | Should adapters log their breakdown, or leave that to the caller? | **Caller logs** — adapters are pure computation |
| O3 | Should `estimate_kv_cache` raise on unknown model_type or silently fallback to GQA? | **Silent fallback** with `logger.warning` |

## 8. Documentation

- [kv-cache-calc-research.md](file:///Users/darvin/Documents/local-eval/docs/kv-cache-calc-research.md) — Source of truth for all KV cache formulas and validation numbers
