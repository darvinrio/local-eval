"""
models/results.py

Classes used for storing Eval results
"""

from datetime import datetime

import msgspec


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
