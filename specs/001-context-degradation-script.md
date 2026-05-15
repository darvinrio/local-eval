# Context-Scale Benchmark Spec

> **Goal**: Measure how TPS (tokens-per-second) and TTFT (time-to-first-token)
> degrade as prompt context size increases, using realistic long-form tasks.
> Fully reproducible, memory-safe, and extensible with new tasks.

---

## 1. What This Measures

| Metric | Description |
|---|---|
| `ttft_ms` | Wall-clock ms from generator creation to first token yielded — pure model prefill |
| `prompt_tps` | Prefill throughput (tokens/sec) as reported by `stream_generate` |
| `generation_tps` | Decode throughput (tokens/sec) as reported by `stream_generate` |
| `tokenizer_time_ms` | Time for `apply_chat_template()` to encode the full prompt — tracked **separately** to surface O(n) tokenizer cost |
| `peak_memory_gb` | Unified memory peak during the run |
| `actual_context_tokens` | Real token count after slicing (should equal target ± 0) |

The benchmark sweeps `actual_context_tokens` across a configurable ladder
(`context_sizes`) and runs each active task at every size.

---

## 2. File Layout

```
local-eval/
├── benchmarks/
│   ├── mlx_bm.py                  # existing — unchanged
│   ├── mlx_ctx_bm.py              # NEW — runner function
│   └── tasks/
│       ├── __init__.py            # exports TASK_REGISTRY
│       ├── base.py                # ContextTask dataclass
│       ├── bug_detection.py       # Task: find bugs in a Python file
│       ├── classifier.py          # Task: rule-based text classifier
│       └── dbt_model.py           # Task: convert raw info → dbt model (seed TBD)
├── models/
│   └── results.py                 # add ContextScaleResult + ContextScaleSweepResult
├── mlx_ctx_bench.py               # NEW — root entrypoint with CONFIG dict
└── docs/
    ├── specs/
    │   └── ctx-scale-benchmark.md # this file
    └── tasks/
        └── TASK_SCHEMA.md         # canonical format for authoring new tasks (for LLMs)
```

---

## 3. Task System

### 3.1 `ContextTask` Dataclass (`benchmarks/tasks/base.py`)

```python
@dataclasses.dataclass
class ContextTask:
    task_id: str             # snake_case unique ID, e.g. "bug_detection"
    task_name: str           # Human-readable, e.g. "Bug Detection"
    description: str         # What the task tests (used in logs and output)
    system_prompt: str       # System role message
    user_prompt_prefix: str  # Instruction placed BEFORE the scaled context
    user_prompt_suffix: str  # Instruction placed AFTER the scaled context (may be "")
    seed_content: str        # Raw text/code that gets repeated to fill context
    seed_content_lang: str   # Hint for display: "python", "sql", "text", "html"
```

The full user message is assembled as:
```
{user_prompt_prefix}

{scaled_context}

{user_prompt_suffix}
```

### 3.2 Task Registry (`benchmarks/tasks/__init__.py`)

```python
TASK_REGISTRY: dict[str, ContextTask] = {
    "bug_detection": BUG_DETECTION_TASK,
    "classifier":    CLASSIFIER_TASK,
    "dbt_model":     DBT_MODEL_TASK,
}
```

**Adding a new task** = create a new file, instantiate `ContextTask`, register
it in `TASK_REGISTRY`. No changes to the runner.

### 3.3 Bundled Tasks

#### `bug_detection`
- **System**: `"You are a senior Python engineer performing a code review."`
- **Prefix**: `"Here is a Python source file. Identify every bug, anti-pattern, and code smell you find:"`
- **Suffix**: `"List each issue with: approximate line reference, issue type, and a one-line explanation."`
- **Seed**: ~300-token self-contained Python module with seeded bugs (wrong
  variable names, off-by-one errors, mutable default args, missing `None`
  checks, shadowed builtins). Repeated to fill target context.
- **Lang**: `python`

#### `classifier`
- **System**: `"You are a strict rule-based text classifier."`
- **Prefix**: `"Here is a list of classification rules followed by a document to classify:"`
- **Suffix**: `"Apply the rules above in order. Output only the matching rule IDs and a one-sentence justification."`
- **Seed**: ~250-token block alternating rule definitions and sample documents.
  Repeated to fill target context.
- **Lang**: `text`

#### `dbt_model` *(starter — seed content from TASK_SCHEMA.md prompt)*
- **System**: `"You are a senior analytics engineer."`
- **Prefix**: `"Here is raw business information, data descriptions, and source table schemas:"`
- **Suffix**: `"Convert the above into a dbt staging model SQL file with appropriate column renaming, type casting, and inline comments."`
- **Seed**: ~300-token block of business entity descriptions, raw column names,
  and data types. Repeated to fill target context.
- **Lang**: `sql`

---

## 4. Context Building (`_build_context`)

**Algorithm (token-ID slicing)**:

```
1. Concatenate seed_content × N until raw string is >> target chars
2. Tokenize the full blob → token_ids (list)
3. Slice token_ids[:target_tokens]
4. Decode sliced token_ids → scaled_context string
5. Assemble full prompt and tokenize once → final actual_context_tokens
```

- Single tokenizer call on the large blob, then one decode, then one final
  encode. Exact to the token count.
- `actual_context_tokens` in results comes from step 5, not the target —
  drift is always visible.

---

## 5. TTFT Measurement

```python
# --- Tokenizer timing (measured separately) ---
t0 = time.perf_counter()
tokenized_prompt = tokenizer.apply_chat_template(messages, ...)
tokenizer_time_ms = (time.perf_counter() - t0) * 1000

# --- Generator TTFT (pure model prefill) ---
gen = stream_generate(model, tokenizer=tokenizer, prompt=tokenized_prompt, max_tokens=max_tokens)

t_start = time.perf_counter()
t_first_token: float | None = None
for response in gen:
    if t_first_token is None:
        t_first_token = time.perf_counter()
    ...

ttft_ms = (t_first_token - t_start) * 1000
```

- Clock starts **at generator creation** (after tokenisation is done).
- Separates tokenizer O(n) cost from model prefill latency — both are visible
  in the output table.

---

## 6. Memory Pre-flight

Before the sweep begins, for each `(task, context_size)` pair:

```
kv_cache_gb = (
    context_tokens × num_layers × num_kv_heads × head_dim × 2 × dtype_bytes
) / (1024**3)

required_gb = model_weights_gb + kv_cache_gb + generation_headroom_gb
available_gb = psutil.virtual_memory().available / (1024**3)
```

`num_layers`, `num_kv_heads`, `head_dim` are read from the model `config` dict
(already loaded via `load(..., return_config=True)`).
`model_weights_gb` is read from `mx.get_active_memory()` immediately
after model load.

### Decision tree

```
required_gb > available_gb × safety_threshold
    AND force_run == False  →  WARN + SKIP
                               (ContextScaleResult.skipped = True, skip_reason set)
    AND force_run == True   →  interactive prompt:
                               "⚠  {context_tokens} tokens may use {required_gb:.1f} GB
                                   (available: {available_gb:.1f} GB). Proceed? [y/N]: "
                               "n" / default → skip
                               "y"           → run anyway
```

---

## 7. Run Structure

```
load model once
  for task in active_tasks:
    for context_size in context_sizes:
      memory_preflight(task, context_size)   # may skip
      scaled_context = build_context(task.seed_content, context_size)
      for _ in range(warmup_runs):           # discarded
          _single_run(...)
      measurements = [_single_run(...) for _ in range(num_runs)]
      result = aggregate(measurements)       # mean across num_runs
unload model
emit JSON + rich table
```

---

## 8. Result Schema (`models/results.py` additions)

```python
class ContextScaleResult(msgspec.Struct):
    task_id: str
    target_context_tokens: int
    actual_context_tokens: int
    tokenizer_time_ms: float          # separate from model TTFT
    ttft_ms: float
    prompt_tps: float                 # mean across num_runs
    generation_tps: float             # mean across num_runs
    peak_memory_gb: float             # mean across num_runs
    generation_tokens: int
    skipped: bool = False
    skip_reason: str = ""


class ContextScaleSweepResult(msgspec.Struct):
    model: str
    run_at: str                       # ISO timestamp
    config: dict[str, Any]           # snapshot of CONFIG dict
    results: list[ContextScaleResult]
```

---

## 9. Output

### JSON
Written to `output/ctx_scale_{model_slug}_{timestamp}.json`.
Serialised via `msgspec.json.encode`.

### Rich Terminal Table

```
Task          | Context Tokens | TTFT (ms) | Tok. Time (ms) | Prefill TPS | Decode TPS | Mem (GB) | Status
bug_detection |            512 |      48.2 |           12.1 |      3821.4 |      118.6 |     8.12 | ✓
bug_detection |           1024 |      89.7 |           18.3 |      3612.0 |      117.9 |     8.88 | ✓
bug_detection |           4096 |     342.1 |           61.4 |      3200.5 |      116.1 |    11.40 | ✓
bug_detection |          16384 |       —   |              — |           — |          — |        — | SKIP
classifier    |            512 |      51.1 |           11.8 |      3799.2 |      119.0 |     8.14 | ✓
...
```

---

## 10. Config Dict (`mlx_ctx_bench.py` `__main__`)

```python
CONFIG = {
    # Model
    "model":   "mlx-community/Qwen3-8B-4bit",
    "seed":    42,

    # Context sweep
    "context_sizes": [512, 1024, 2048, 4096, 8192, 16384, 32768],

    # Tasks — must be keys registered in TASK_REGISTRY
    "active_tasks": ["bug_detection", "classifier"],

    # Generation
    "max_tokens": 256,

    # Run control
    "warmup_runs": 1,
    "num_runs":    3,

    # Memory safety
    "memory_safety_threshold": 0.85,   # fraction of available RAM
    "force_run":               False,  # True → interactive override prompt
    "generation_headroom_gb":  1.0,    # buffer on top of weight + KV estimate

    # Output
    "output_dir": "output",
}
```

---

## 11. TASK_SCHEMA.md — Prompt Format for LLM Task Authoring

See `docs/tasks/TASK_SCHEMA.md`. The schema document:
- Specifies all required fields with types and constraints
- Gives a worked example (`bug_detection`)
- States seed content guidelines (target length, realistic content, no
  degenerate repetition within the seed itself)
- Is designed to be copy-pasted as a prompt to any capable LLM to produce a
  valid new `ContextTask` module

---

## 12. Open Items / Future Work

- [ ] Degrade curve plots: TTFT vs context_size per task (polars + matplotlib)
- [ ] Multi-model outer loop
- [ ] Add `prompt_tps_stddev` / `generation_tps_stddev` when `num_runs > 1`
- [ ] PPL-at-context variant
- [ ] Export results as Polars DataFrame → Parquet
