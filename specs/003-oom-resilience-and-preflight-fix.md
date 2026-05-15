# Spec 003 — OOM Resilience & Preflight Config Fix

> **Context**: Running `mlx_ctx_bench` on two ~17 GB models
> (`Jiunsong/supergemma4-26b-uncensored-mlx-4bit-v2` and
> `unsloth/Qwen3.6-35B-A3B-UD-MLX-3bit`) crashed with a Metal OOM at large
> context sizes. The preflight check should have caught this but silently
> passed. Two independent but related issues.

---

## Issue 1 — Graceful Exception Handling (OOM & General)

### Problem

When the Metal GPU runs out of memory during inference, the MLX runtime
throws an uncatchable `std::runtime_error` that propagates as a Python
`RuntimeError`:

```
libc++abi: terminating due to uncaught exception of type std::runtime_error:
[METAL] Command buffer execution failed: Insufficient Memory
(00000008:kIOGPUCommandBufferCallbackErrorOutOfMemory)
```

The current code has a `try/finally` around the full sweep loop (line 273 of
`mlx_ctx_bm.py`), but its only purpose is to call `unload()`. When the
exception fires mid-generation, **all results collected so far are lost**
because `run_ctx_sweep` never returns — the exception propagates up to
`main()` which has no handler either.

### Root Cause

No `except` clause around the per-iteration work. The crash during a large
context size (e.g. 131072 tokens) kills the process before the function can
return the `results` list that already contains successful entries for smaller
context sizes.

### Spec

#### 1.1 Per-iteration exception guard

Wrap each `(task, context_size)` iteration body in a `try/except` that
catches `Exception` (which includes `RuntimeError` from Metal OOM, plus any
other failure during generation):

```python
for task in tasks:
    for context_size in config.context_sizes:
        try:
            # ... existing: _build_context, _memory_preflight, warmup, measure, aggregate
        except Exception as exc:
            logger.error(
                f"Task {task.task_id} @ {context_size} FAILED: {exc}"
            )
            results.append(
                ContextScaleResult(
                    task_id=task.task_id,
                    target_context_tokens=context_size,
                    actual_context_tokens=actual_tokens if 'actual_tokens' in dir() else 0,
                    tokenizer_time_ms=0,
                    ttft_ms=0,
                    prompt_tps=0,
                    generation_tps=0,
                    peak_memory_gb=0,
                    generation_tokens=0,
                    skipped=True,
                    skip_reason=f"error: {exc}",
                )
            )
            # Clear GPU caches to attempt recovery for subsequent sizes
            mx.clear_cache()
            gc.collect()
            continue
```

#### 1.2 New `ContextScaleResult` status semantics

Currently `skipped` is a bool used only for preflight skips. Extend the
result model to distinguish causes:

| `skipped` | `skip_reason` prefix | Meaning |
|-----------|---------------------|---------|
| `False` | `""` | Completed normally |
| `True` | `"Required memory..."` | Preflight skip |
| `True` | `"error: ..."` | Runtime error (OOM, generation failure, etc.) |

> **Note**: A future iteration could replace the `skipped` bool with a proper
> enum (`completed`, `preflight_skip`, `error`), but for now keeping
> `skipped + skip_reason` prefix is backward-compatible with existing output
> consumers.

#### 1.3 Rich table status column

Add a third status indicator in `emit_rich_table`:

```
[green]✓[/green]     → completed
[yellow]SKIP[/yellow] → preflight skip (skip_reason does NOT start with "error:")
[red]ERR[/red]       → runtime error  (skip_reason starts with "error:")
```

#### 1.4 Early persist in `main()`

Move the JSON write **inside** the `finally` block or after the sweep call
returns, so partial results are always saved. Currently the write happens
after `run_ctx_sweep` returns, which is fine as long as 1.1 prevents the
exception from propagating. No structural change needed if 1.1 is
implemented correctly, but add a defensive note:

```python
sweep_result = run_ctx_sweep(CONFIG, tasks_to_run)
# run_ctx_sweep now always returns — errors are recorded as skipped results
```

#### 1.5 Metal OOM recovery caveat

> **Warning**: Some Metal OOM errors are truly fatal and may corrupt the MLX
> allocator state. After catching one, subsequent runs at *any* context size
> may also fail. The `mx.clear_cache()` + `gc.collect()` call is best-effort.
> If recovery fails, the remaining iterations will also be caught and recorded
> as errors.

Consider adding a `_consecutive_error_limit` (default: 3). If N consecutive
iterations within the same task fail with exceptions, log a warning and
`break` out of the context_sizes loop for that task (move to next task).

---

## Issue 2 — Preflight Config Key Resolution

### Problem

`_memory_preflight` reads architecture parameters directly from the config
dict returned by `mlx_lm.load(..., return_config=True)`:

```python
num_layers = config.get("num_hidden_layers", config.get("num_layers", 0))
num_kv_heads = config.get("num_key_value_heads", config.get("num_kv_heads", 0))
head_dim = config.get("head_dim", ...)
```

This works for **text-only models** (e.g. `Qwen3.5-9B`) where these keys
live at the config root. But for **multimodal / conditional-generation
models**, `mlx_lm.load` returns the raw `config.json` dict where these keys
are nested under `text_config`.

### Root Cause — Config Key Nesting

**SuperGemma4** (`Gemma4ForConditionalGeneration`):
```json
{
  "model_type": "gemma4",
  "text_config": {
    "num_hidden_layers": 30,
    "num_key_value_heads": 8,
    "num_attention_heads": 16,
    "head_dim": 256,
    "global_head_dim": 512,
    "hidden_size": 2816
  }
}
// Top-level: NO num_hidden_layers, NO num_key_value_heads, etc.
```

**Qwen3.6-35B-A3B** (`Qwen3_5MoeForConditionalGeneration`):
```json
{
  "model_type": "qwen3_5_moe",
  "text_config": {
    "num_hidden_layers": 40,
    "num_key_value_heads": 2,
    "num_attention_heads": 16,
    "head_dim": 256,
    "hidden_size": 2048,
    "linear_key_head_dim": 128,
    "linear_num_key_heads": 16,
    "linear_num_value_heads": 32,
    "linear_value_head_dim": 128
  }
}
// Top-level: NO num_hidden_layers, NO num_key_value_heads, etc.
```

**Result**: Every `.get()` falls through to default `0`. The formula
computes `kv_cache_gb = 0`. The preflight comparison becomes:

```
required_gb = model_weights_gb + 0 + 1.0  (headroom)
            ≈ 17 + 0 + 1 = 18 GB
```

This always passes against `pre_load_available_gb * 0.95 ≈ 38 GB` on a
48 GB machine, even when the real KV cache at 131k tokens would need
~19 GB (SuperGemma4) or ~5 GB (Qwen3.6).

### Spec

#### 2.1 Config resolution function

Create a helper that resolves the text/language model config regardless of
nesting:

```python
def _resolve_text_config(config: dict[str, Any]) -> dict[str, Any]:
    """
    Extract the text model config from a potentially nested model config.

    Multimodal models (Gemma4ForConditionalGeneration,
    Qwen3_5MoeForConditionalGeneration, etc.) nest architecture params
    under 'text_config' or 'language_config'. Text-only models have them
    at the root level.

    Args:
        config: Raw config dict from mlx_lm.load(..., return_config=True).

    Returns:
        The sub-dict containing num_hidden_layers, num_key_value_heads, etc.
    """
    for sub_key in ("text_config", "language_config", "llm_config"):
        if sub_key in config and isinstance(config[sub_key], dict):
            sub = config[sub_key]
            if "num_hidden_layers" in sub or "num_layers" in sub:
                return sub
    return config
```

**Call site**: At the top of `run_ctx_sweep`, once, before the loop so the
resolved config is reused:

```python
text_config = _resolve_text_config(model_config)
# Then pass text_config to _memory_preflight instead of model_config
```

#### 2.2 Mixed attention type awareness

Both models use heterogeneous layer types that affect KV cache size:

| Model | Layer Types | KV Cache Difference |
|-------|------------|-------------------|
| SuperGemma4 | 25× `sliding_attention` + 5× `full_attention` | `sliding` uses `head_dim=256`, `full` uses `global_head_dim=512` |
| Qwen3.6 | 30× `linear_attention` + 10× `full_attention` | `linear` uses `linear_key_head_dim=128` / `linear_num_key_heads=16`; `full` uses `head_dim=256` / `num_key_value_heads=2` |

The current formula assumes uniform layers:
```
kv_cache_gb = tokens × num_layers × num_kv_heads × head_dim × 2 × dtype_bytes
```

**Proposed approach — conservative upper-bound**:

Rather than attempting to model each architecture's layer heterogeneity
exactly (which is fragile and architecture-specific), use a **worst-case
estimate** by computing per-layer KV cache using the *largest* head_dim and
*largest* num_kv_heads found in the config:

```python
head_dim = max(
    text_config.get("head_dim", 0),
    text_config.get("global_head_dim", 0),
    text_config.get("linear_key_head_dim", 0),
    text_config.get("linear_value_head_dim", 0),
)

num_kv_heads = max(
    text_config.get("num_key_value_heads", 0),
    text_config.get("num_global_key_value_heads", 0),
    text_config.get("linear_num_key_heads", 0),
    text_config.get("linear_num_value_heads", 0),
)
```

This intentionally overestimates for safety. The preflight is a guard, not
a precise calculator — false positives (skipping a run that would have
succeeded) are far preferable to false negatives (OOM crash losing all data).

#### 2.3 Zero-estimate guard

Add a hard check after resolving config values. If the KV cache estimate
comes out to zero (because config keys are missing or unrecognized), log a
warning and apply a conservative fallback:

```python
if kv_cache_gb == 0 and context_tokens > 0:
    logger.warning(
        f"KV cache estimate is 0 GB for {context_tokens} tokens. "
        f"Config may be missing expected keys. "
        f"Falling back to heuristic: 0.5 bytes/token/layer."
    )
    num_layers_fallback = text_config.get(
        "num_hidden_layers", text_config.get("num_layers", 32)
    )
    # Heuristic: ~0.5 bytes per token per layer is a rough lower bound
    # for GQA models with small KV heads
    kv_cache_gb = (context_tokens * num_layers_fallback * 0.5) / (1024**3)
```

> **Important**: The zero-estimate guard must also log the resolved config
> keys so the operator can diagnose *why* the estimate was zero. This
> prevents silent preflight bypass from ever happening again.

#### 2.4 Preflight diagnostic logging

On every preflight call, log the resolved values at `DEBUG` level:

```python
logger.debug(
    f"Preflight: layers={num_layers} kv_heads={num_kv_heads} "
    f"head_dim={head_dim} → kv_cache={kv_cache_gb:.2f} GB, "
    f"model_weights={model_weights_gb:.2f} GB, "
    f"required={required_gb:.2f} GB vs "
    f"available={pre_load_available_gb:.2f} GB × {safety_threshold}"
)
```

This makes future misestimations immediately visible in logs.

---

## Worked Examples

### SuperGemma4-26b-4bit @ 131072 tokens (corrected preflight)

```
text_config:
  num_hidden_layers = 30
  num_key_value_heads = 8     (but also num_global_key_value_heads = 2)
  head_dim = 256              (but also global_head_dim = 512)
  
Conservative max:
  head_dim = max(256, 512) = 512
  num_kv_heads = max(8, 2) = 8

kv_cache_gb = 131072 × 30 × 8 × 512 × 2 × 2 / (1024**3)
            = 131072 × 30 × 8 × 512 × 4 / (1024**3)
            ≈ 60 GiB

model_weights ≈ 17 GiB
headroom = 1.0 GiB
required = 17 + 60 + 1.0 = 78 GiB

On 48 GB machine: 78 > 38 × 0.95 → SKIP ✓
```

(The real KV cache is smaller due to sliding attention layers using the
smaller head_dim, but the conservative estimate correctly blocks the run.)

### Qwen3.6-35B-A3B-3bit @ 65536 tokens (corrected preflight)

```
text_config:
  num_hidden_layers = 40
  num_key_value_heads = 2    (full_attention layers)
  linear_num_key_heads = 16  (linear_attention layers)
  head_dim = 256             (full_attention)
  linear_key_head_dim = 128  (linear_attention)
  linear_num_value_heads = 32

Conservative max:
  head_dim = max(256, 128, 128) = 256
  num_kv_heads = max(2, 16, 32) = 32

kv_cache_gb = 65536 × 40 × 32 × 256 × 2 × 2 / (1024**3)
            ≈ 75 GiB → SKIP ✓

(Highly conservative — real usage is lower because linear attention layers
don't maintain traditional KV caches. But the estimate correctly prevents
the crash.)
```

---

## Files Changed

| File | Change |
|------|--------|
| `benchmarks/mlx_ctx_bm.py` | Add `_resolve_text_config()`, refactor `_memory_preflight()` to use resolved config, add per-iteration try/except, add `mx.clear_cache()` on error, add zero-estimate guard, add diagnostic logging |
| `mlx_ctx_bench.py` | Add `ERR` status to `emit_rich_table` |
| `models/results.py` | No schema changes needed (existing `skipped` + `skip_reason` fields are sufficient) |

---

## Open Questions

1. **Consecutive error bail-out threshold**: Should the runner stop trying
   larger context sizes for a given task after N consecutive errors? Proposed
   default: 3. This avoids N×timeout for inevitably-failing sizes after OOM.

2. **Conservative overestimate tolerance**: The worst-case `max()` approach
   for mixed-attention models may skip context sizes that would actually fit.
   Is this acceptable, or should we invest in per-architecture KV cache
   calculators? (Recommendation: accept the overestimate for now; it's a
   safety guard, not a capacity planner.)

3. **Fatal vs. recoverable OOM**: Should we attempt to detect truly fatal
   Metal errors (process-level corruption) and abort the entire sweep vs.
   just the current iteration? This may not be reliably detectable from
   Python.
