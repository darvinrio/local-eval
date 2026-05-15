"""
models/config.py

Config Variables
"""

from typing import Final, Literal

import msgspec


class Config(msgspec.Struct):
    """Config storing variables for benchmarking"""

    LATENCY_PROMPT: str


ALLOWED_STRIDES: Final[tuple[int, ...]] = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512)
Stride = Literal[1, 2, 4, 8, 16, 32, 64, 128, 256, 512]


class MLXContextConfig(msgspec.Struct):
    """
    Config storing variables for context benchmarking

    Args:
        model_name: Name of the model to use.
        seed: Random seed for reproducibility.
        context_sizes: List of context sizes to test.
        active_tasks: List of tasks to run.
        num_runs: Number of runs per task.
        max_tokens: Maximum number of tokens to generate.
        warmup_runs: Number of warmup runs.
        memory_safety_threshold: Fraction of available RAM to use.
        force_run: Whether to force run even if memory is insufficient.
        generation_headroom_gb: Headroom for generation in GiB.
        output_dir: Directory to save results.
        capture_per_token_timings: Whether to capture per-token timings.
        per_token_timing_max_tokens: Maximum number of tokens to capture per run.
        trace_stride: Stride for capturing per-token timings.
        include_final_token_in_trace: Whether to include the final token in the trace.

    Note:
        `trace_stride` is a power of 2 to ensure efficient sampling and not zero.
    """

    model_name: str
    seed: int
    context_sizes: list[int]
    active_tasks: list[str]
    num_runs: int
    max_tokens: int
    warmup_runs: int
    memory_safety_threshold: float
    force_run: bool
    generation_headroom_gb: float
    output_dir: str
    capture_per_token_timings: bool = True
    per_token_timing_max_tokens: int | None = None
    trace_stride: Stride = 32
    include_final_token_in_trace: bool = True
    kv_quant_bits: int = 16

    def __post_init__(self) -> None:
        """Validate the stride is one of the allowed values."""
        if self.trace_stride not in ALLOWED_STRIDES:
            raise ValueError(f"stride must be one of {ALLOWED_STRIDES}")
