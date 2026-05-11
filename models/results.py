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


class ContextScaleSweepResult(msgspec.Struct):
    """ContextScaleSweepResult"""

    model: str
    run_at: str  # ISO timestamp
    config: MLXContextConfig  # snapshot of CONFIG dict
    results: list[ContextScaleResult]
