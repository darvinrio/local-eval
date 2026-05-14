# 004 — KV Cache Adapter Refactor

**Status**: Draft
**Created**: 2026-05-13
**Related**: `specs/003-oom-resilience-and-preflight-fix.md`, `docs/kv-cache-calc-approx.md`

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
    adapters.py          # Gemma4, Qwen35Moe, StandardGQA + registry
```

### 3.1 Data Models (`models.py`)

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class LayerKVCacheInfo:
    """KV cache cost for one group of same-type layers."""
    layer_type: str                  # "sliding_attention", "full_attention", "linear_attention", "gqa", ...
    num_layers: int
    bytes_per_token_per_layer: int   # logical bytes one new token adds per layer
    max_cache_tokens: int | None     # None = unbounded; else sliding-window cap

@dataclass
class KVCacheEstimate:
    """Full-model KV cache estimate with per-layer-type breakdown."""
    model_type: str
    kv_quant_bits: int               # effective bits used (16, 8, 4)
    layer_breakdowns: list[LayerKVCacheInfo]

    @property
    def bytes_per_token(self) -> int:
        """Total bytes/token across all layer types (uncapped)."""
        return sum(
            lb.num_layers * lb.bytes_per_token_per_layer
            for lb in self.layer_breakdowns
        )

    def estimate_bytes(self, context_tokens: int) -> int:
        """Total KV cache bytes for a given context length, with sliding-window capping."""
        total = 0
        for lb in self.layer_breakdowns:
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
- `layer_types` list -> count `sliding_attention` vs `full_attention`
- `num_key_value_heads`, `head_dim` -> for sliding layers
- `num_global_key_value_heads`, `global_head_dim` -> for global layers
- `attention_k_eq_v` -> if `True`, global layers use unified K=V (1 tensor)
- `sliding_window` -> cap for sliding layers

**Formulas** (per the reference doc):

| Layer type | Formula (bytes/layer/token) |
|---|---|
| `sliding_attention` | `2 x num_kv_heads x head_dim x dtype_bytes` |
| `full_attention` (K=V unified) | `1 x num_global_kv_heads x global_head_dim x dtype_bytes` |
| `full_attention` (K!=V) | `2 x num_global_kv_heads x global_head_dim x dtype_bytes` |

Where `dtype_bytes = kv_quant_bits / 8`.

Sliding layers get `max_cache_tokens = sliding_window`. Full-attention layers are unbounded (`None`).

**Validation against reference doc** (Gemma4 26B-A4B, bf16):
```
sliding_bpt_layer = 2 x 8 x 256 x 2 = 8,192 bytes/token/layer
global_bpt_layer  = 1 x 2 x 512 x 2 = 2,048 bytes/token/layer (unified K=V)

bytes_per_token = 25 x 8,192 + 5 x 2,048 = 204,800 + 10,240 = 215,040 = 210 KiB/token  ✓
```

#### 3.3.2 `Qwen35MoeAdapter`

**Supported model_types**: `"qwen3_5_moe"`, `"qwen3_5_moe_text"`

Reads from config:
- `layer_types` list -> count `full_attention` vs `linear_attention`
- `num_key_value_heads`, `head_dim` -> for full_attention layers
- `linear_num_key_heads`, `linear_key_head_dim` -> K dims for linear layers
- `linear_num_value_heads`, `linear_value_head_dim` -> V dims for linear layers

**Formulas**:

| Layer type | Formula (bytes/layer/token) |
|---|---|
| `full_attention` | `2 x num_kv_heads x head_dim x dtype_bytes` |
| `linear_attention` | `(linear_num_key_heads x linear_key_head_dim + linear_num_value_heads x linear_value_head_dim) x dtype_bytes` |

Both types are unbounded (`max_cache_tokens = None`). No sliding window in this architecture.

**Validation** (Qwen3.6 35B-A3B, bf16):
```
full_attn:   10 layers x (2 x 2 x 256 x 2)             = 10 x 2,048  = 20,480 B/token
linear_attn: 30 layers x ((16x128 + 32x128) x 2)        = 30 x 12,288 = 368,640 B/token
total = 389,120 B/token ~ 380 KiB/token
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

    model_weights_gb = mx.get_active_memory() / 1e9
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
| `utils/kv_cache/adapters.py` | Create | `Gemma4Adapter`, `Qwen35MoeAdapter`, `StandardGQAAdapter`, registry |
| `benchmarks/mlx_ctx_bm.py` | Modify | Replace KV math in `_memory_preflight` with adapter call; add `kv_quant_bits` param |
| `models/config.py` | Modify | Add `kv_quant_bits: int = 16` to `MLXContextConfig` |

## 6. Testing Strategy

- **Unit tests** for each adapter using the sample configs in `output/samples/`.
- Validate against the reference numbers in `docs/kv-cache-calc-approx.md`:
  - Gemma4 26B-A4B: 210 KiB/token (bf16)
  - Qwen3 8B (standard GQA fallback): 144 KiB/token
- Test `kv_quant_bits` override: e.g., Gemma4 at 8-bit should yield exactly half of bf16.
- Test sliding-window capping: at `context_tokens < sliding_window`, result matches uncapped; at `context_tokens > sliding_window`, sliding layers are capped.

## 7. Open Decisions

| # | Question | Default |
|---|---|---|
| O1 | Should `_resolve_text_config` also move to `utils/kv_cache/`? | **No** — keep in benchmark; adapters receive already-resolved config |
| O2 | Should adapters log their breakdown, or leave that to the caller? | **Caller logs** — adapters are pure computation |
| O3 | Should `estimate_kv_cache` raise on unknown model_type or silently fallback to GQA? | **Silent fallback** with `logger.warning` |
