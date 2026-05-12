"""
models/results.py

Classes used for storing Eval results
"""

from datetime import datetime

import msgspec

from models.config import MLXContextConfig


class BenchmarkResult(msgspec.Struct):
    """BenchmarkResult"""

    prompt_tps: float  # prefill
    generation_tps: float  # decode
    peak_memory_gb: float  # unified memory peak
    prompt_tokens: int
    generation_tokens: int


class EvalResult(msgspec.Struct):
    """EvalResult"""

    task: str
    accuracy: float
    stderr: float


class PPLResult(msgspec.Struct):
    """PPLResult"""

    ppl: float
    corpus: str


class ModelResult(msgspec.Struct):
    """ModelResult"""

    model: str
    run_at: str = msgspec.field(
        default_factory=lambda: datetime.now().isoformat(timespec="seconds")
    )
    benchmarks: list[BenchmarkResult] = []
    evals: list[EvalResult] = []
    ppls: list[PPLResult] = []
    errors: list[str] = []


class TokenTiming(msgspec.Struct):
    """Per-token timing from an individual benchmark run."""

    token_index: int  # 1-based generated token position
    delta_ms: float  # wall-clock time since previous yielded token
    since_start_ms: float  # wall-clock time since decode start
    since_first_token_ms: float  # wall-clock time since first yielded token
    sequence_length: int  # prompt_tokens + token_index


class ContextScaleRunResult(msgspec.Struct):
    """Raw metrics for a single measured benchmark run."""

    run_index: int
    tokenizer_time_ms: float
    ttft_ms: float
    prompt_tps: float
    generation_tps: float
    peak_memory_gb: float
    generation_tokens: int
    prompt_tokens: int
    per_token_timings: list[TokenTiming] = []


class ContextScaleResult(msgspec.Struct):
    """ContextScaleResult"""

    task_id: str
    target_context_tokens: int
    actual_context_tokens: int
    tokenizer_time_ms: float  # separate from model TTFT
    ttft_ms: float
    prompt_tps: float  # mean across num_runs
    generation_tps: float  # mean across num_runs
    peak_memory_gb: float  # mean across num_runs
    generation_tokens: int
    skipped: bool = False
    skip_reason: str = ""
    runs: list[ContextScaleRunResult] = []


class ContextScaleSweepResult(msgspec.Struct):
    """ContextScaleSweepResult"""

    model: str
    run_at: str  # ISO timestamp
    config: MLXContextConfig  # snapshot of CONFIG dict
    results: list[ContextScaleResult]
