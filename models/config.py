"""
models/config.py

Config Variables
"""

import msgspec


class Config(msgspec.Struct):
    """Config storing variables for benchmarking"""

    LATENCY_PROMPT: str


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
        generation_headroom_gb: Headroom for generation in GB.
        output_dir: Directory to save results.
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
    # Per-token trace capture
    capture_per_token_timings: bool = True
    per_token_timing_max_tokens: int | None = None
    trace_stride: int = 1
    include_final_token_in_trace: bool = True
    rolling_tps_window: int = 32
