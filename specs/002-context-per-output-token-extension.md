# MLX Context Benchmark Extension Spec

## Goal
Extend the existing context-scaling benchmark so it can also capture **per-output-token timing** during generation, store that data in JSON, and preserve the current aggregate benchmark outputs.

The implementation should let downstream analysis answer questions like:
- How long did token 1, 2, 3, ... N take to arrive?
- Does decode latency increase as generated tokens accumulate?
- How does that interact with prompt context size?

The current scripts already measure:
- tokenizer time
- TTFT
- prompt TPS
- aggregate generation TPS
- peak memory
- generation token count

The new implementation must add **decode-growth instrumentation** without breaking the existing context-scale benchmark workflow.

***

## Existing Files
The model should modify the existing implementation centered around:
- `mlx_ctx_bm.py`
- `mlx_ctx_bench.py`

The benchmark currently:
1. builds a prompt for a task and target context size
2. tokenizes it
3. calls `stream_generate(...)`
4. iterates over yielded responses
5. stores only aggregate metrics from the final response

The current generator loop is the correct insertion point for per-token timing.

***

## Design Requirements

### 1. Keep one benchmark system, but two metric layers
Do **not** split this into a separate unrelated script unless absolutely necessary.

Instead, keep a single benchmark harness with:
- **context-scale metrics**: current TTFT / tokenizer time / prompt TPS / generation TPS / memory
- **decode-growth metrics**: per-token timing trace during generation

This means the same run should be able to emit both:
- aggregate metrics for summary tables
- detailed per-token timing suitable for plotting later

The aggregate benchmark behavior must remain intact.

***

### 2. Per-token timing must be recorded for every yielded token
For every generated output token, store timing information.

At minimum record, for token index `i`:
- `token_index`: 1-based generated token position
- `delta_ms`: wall-clock time since the previous yielded token
- `since_start_ms`: wall-clock time since decode start
- `since_first_token_ms`: wall-clock time since the first token was yielded
- `sequence_length`: total active sequence length at this step = prompt token count + generated tokens so far

Definitions:
- `decode start` = timestamp taken immediately before consuming the generator
- `first token time` = timestamp when first yielded token arrives
- `delta_ms` for token 1 should be equal to TTFT or stored explicitly as first-step latency, depending on implementation choice

Preferred behavior:
- token 1 `delta_ms` should represent time from decode start to first token
- token N `delta_ms` should represent time from token N-1 arrival to token N arrival

This gives a clean next-token latency series.

***

### 3. Preserve TTFT semantics
The current TTFT definition must remain unchanged:
- TTFT is measured from generator consumption start to first yielded token
- tokenizer time remains separate

So:
- `tokenizer_time_ms` stays independent
- `ttft_ms` remains a top-level aggregate metric
- per-token trace should be consistent with `ttft_ms`

Recommended consistency rule:
- `per_token_trace[0].delta_ms ~= ttft_ms`

***

### 4. Store per-token traces in JSON output
The JSON output must remain easy to analyze programmatically.

Add a per-run structure rather than only storing averaged aggregate metrics.

Recommended schema approach:

## Schema changes

### Add a per-token timing struct
Use a structured type, e.g.:

```python
class TokenTiming(msgspec.Struct):
    token_index: int
    delta_ms: float
    since_start_ms: float
    since_first_token_ms: float
    sequence_length: int
```

### Add a per-run struct
Store raw run-level measurements so that repeated runs are analyzable:

```python
class ContextScaleRunResult(msgspec.Struct):
    run_index: int
    tokenizer_time_ms: float
    ttft_ms: float
    prompt_tps: float
    generation_tps: float
    peak_memory_gb: float
    generation_tokens: int
    prompt_tokens: int
    per_token_timings: list[TokenTiming]
```

### Update the existing result struct
Keep the current top-level result object for compatibility, but extend it:

```python
class ContextScaleResult(msgspec.Struct):
    task_id: str
    target_context_tokens: int
    actual_context_tokens: int
    tokenizer_time_ms: float
    ttft_ms: float
    prompt_tps: float
    generation_tps: float
    peak_memory_gb: float
    generation_tokens: int
    skipped: bool = False
    skip_reason: str = ""
    runs: list[ContextScaleRunResult] = []
```

Important:
- top-level aggregate fields remain averages across `num_runs`
- `runs` stores raw run-level data
- per-token traces must be stored inside each run, not flattened into the top-level result

Do not remove existing fields used by the terminal table.

***

### 5. Add optional trace controls to config
The benchmark should not always be forced to emit huge trace files.

Add config fields such as:

```python
capture_per_token_timings: bool = True
per_token_timing_max_tokens: int | None = None
rolling_tps_window: int = 32
```

Behavior:
- `capture_per_token_timings=False` disables detailed tracing entirely
- if `per_token_timing_max_tokens` is set, only capture trace up to that many generated tokens even if generation continues
- aggregate metrics should still reflect the full generation run

If simpler, `per_token_timing_max_tokens` may default to `max_tokens`.

***

### 6. Support very long generations safely
The user may later increase `max_tokens` substantially, even to tens of thousands.

The implementation must be written so it does not become awkward at large output lengths.

Requirements:
- storing 256 token timings is trivial
- storing 64k timings should still work in JSON, though large
- code should avoid unnecessary copies or string accumulation for trace capture
- trace capture should append lightweight structs only

Optional but recommended:
- support a `trace_stride` config so the benchmark can store every token or every Nth token for very long runs

Example:
```python
trace_stride: int = 1
```

Behavior:
- `1` = record every token
- `8` = record token 1, 9, 17, ... plus optionally final token

If implemented, document clearly whether skipped tokens affect aggregate metrics (they should not).

***

### 7. Add derived rolling decode stats if easy
If implementation cost is low, also compute optional derived stats per stored token:
- `rolling_tps`
- `rolling_delta_ms_mean`

These are convenience fields for plotting and can reduce downstream analysis effort.

If added, they should be based only on observed/stored timings and clearly documented.

Do not block implementation on this; core requirement is raw per-token timing.

***

## Required code changes

### A. Update `_single_run(...)`
Refactor `_single_run(...)` so it returns a structured run result instead of only a tuple.

Current behavior:
- tokenizes prompt
- starts generator
- loops through responses
- captures first token time
- returns only aggregate metrics from final response

New behavior:
- keep tokenizer timing and TTFT logic
- iterate over every yielded response
- timestamp each yield using `time.perf_counter()`
- build `per_token_timings` list during generation
- preserve final response aggregate metrics
- return a structured object containing both aggregate and per-token data

Implementation notes:
- take `prompt_tokens = len(tokenized_prompt)` once
- set `t_start = time.perf_counter()` immediately before entering the generator loop
- `prev_time = t_start`
- `t_first_token = None`
- for each yielded token:
  - `now = time.perf_counter()`
  - if first token: set `t_first_token = now`
  - compute `delta_ms = (now - prev_time) * 1000`
  - compute `since_start_ms = (now - t_start) * 1000`
  - compute `since_first_token_ms = 0.0 if first token else (now - t_first_token) * 1000`
  - compute `sequence_length = prompt_tokens + token_index`
  - append `TokenTiming(...)`
  - set `prev_time = now`

The loop should still keep the last `response` object so aggregate fields can be read from it.

***

### B. Update aggregation logic in `run_ctx_sweep(...)`
The sweep currently aggregates tuples. That must be updated to aggregate structured run results.

Required behavior:
- warmup runs do not need to be stored in output
- measured runs must be stored in `runs`
- top-level fields must be computed as averages over measured runs
- `generation_tokens` can remain from the final measured run, or preferably use mean/int-rounded behavior if variable across runs

Preferred aggregation:
- average tokenizer time
- average TTFT
- average prompt TPS
- average generation TPS
- average peak memory
- generation token count from last run only if deterministic; otherwise store the per-run variation in `runs`

Do not average `per_token_timings` across runs inside the benchmark unless explicitly adding a second derived summary object.
Raw traces per run are more useful and more lossless.

***

### C. Keep terminal table simple
Do not try to print per-token timing in the terminal table.

The rich table in `mlx_ctx_bench.py` should remain summary-only, using aggregate fields.

This benchmark should output:
- concise terminal summary
- detailed JSON for later plotting

That separation is intentional.

***

### D. Keep backward compatibility where practical
Minimize breakage for existing analysis code.

Requirements:
- existing top-level fields should still exist
- output JSON filename pattern can stay unchanged
- existing benchmark flow should still work with no changes other than schema additions

If any field names must change for consistency, do so only if clearly justified and update all call sites.

***

## Recommended config additions
Extend `MLXContextConfig` with the following or equivalent fields:

```python
capture_per_token_timings: bool = True
per_token_timing_max_tokens: int | None = None
trace_stride: int = 1
include_final_token_in_trace: bool = True
```

Semantics:
- `capture_per_token_timings`: enable/disable trace storage
- `per_token_timing_max_tokens`: cap how many generated tokens are stored in trace
- `trace_stride`: sampling stride for stored token timings
- `include_final_token_in_trace`: ensures last token is captured even if it does not align with stride

These settings only affect stored trace granularity, not actual generation behavior.

***

## JSON example
The resulting JSON for one task/context pair should look conceptually like:

```json
{
  "task_id": "bug_detection",
  "target_context_tokens": 4096,
  "actual_context_tokens": 4108,
  "tokenizer_time_ms": 61.4,
  "ttft_ms": 342.1,
  "prompt_tps": 3200.5,
  "generation_tps": 116.1,
  "peak_memory_gb": 11.40,
  "generation_tokens": 256,
  "skipped": false,
  "skip_reason": "",
  "runs": [
    {
      "run_index": 1,
      "tokenizer_time_ms": 60.9,
      "ttft_ms": 338.7,
      "prompt_tps": 3214.8,
      "generation_tps": 117.2,
      "peak_memory_gb": 11.35,
      "generation_tokens": 256,
      "prompt_tokens": 4108,
      "per_token_timings": [
        {
          "token_index": 1,
          "delta_ms": 338.7,
          "since_start_ms": 338.7,
          "since_first_token_ms": 0.0,
          "sequence_length": 4109
        },
        {
          "token_index": 2,
          "delta_ms": 8.5,
          "since_start_ms": 347.2,
          "since_first_token_ms": 8.5,
          "sequence_length": 4110
        }
      ]
    }
  ]
}
```

Exact numbers do not matter; structure does.

***

## Plotting expectations
The JSON should make it easy to plot later with Python/Polars/Pandas.

At minimum, downstream analysis should be able to derive:
- token index vs `delta_ms`
- token index vs `since_start_ms`
- sequence length vs `delta_ms`
- context size vs TTFT
- context size vs aggregate generation TPS
- context size vs token-N latency

This is why raw per-run traces are required.

***

## Sanity and interpretation notes
The implementation should include comments or docstrings clarifying that:
- per-token timing is an end-to-end token arrival metric, not a pure kernel-time metric
- it reflects real observed streaming latency
- it is useful for decode-growth analysis
- it should not replace the context-scale benchmark, but complement it

The benchmark is intended to answer two different questions:
1. How does prompt length affect TTFT and aggregate throughput?
2. How does decode latency evolve as more output tokens are generated?

The system should support both cleanly.

***

## Acceptance criteria
The change is complete when all of the following are true:

1. Running the existing benchmark still prints a summary table.
2. JSON output still includes existing top-level aggregate metrics.
3. JSON output now also includes raw per-run per-token timing traces.
4. Token 1 timing is approximately equal to TTFT.
5. Each stored token timing includes sequence length and token index.
6. Warmup runs are excluded from stored results.
7. Trace capture can be disabled by config.
8. The implementation remains readable and minimal, not a rewrite of the whole benchmark.

***

## Implementation style guidance
Please keep the patch surgical and maintainable.

Guidelines:
- prefer small dataclass/msgspec additions over major architectural changes
- keep function names close to current naming unless consistency requires renaming
- avoid introducing plotting code into the benchmark itself
- keep JSON generation in the existing output flow
- do not add unrelated features

The deliverable is updated benchmark code that emits detailed per-token timing traces in JSON while preserving current aggregate benchmark behavior.