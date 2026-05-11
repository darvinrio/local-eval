"""
mlx_ctx_bench.py

Context scaling benchmark root entrypoint.
"""

import os
import time
from typing import Any

import msgspec
from rich.console import Console
from rich.table import Table

from benchmarks.mlx_ctx_bm import run_ctx_sweep
from benchmarks.tasks import TASK_REGISTRY
from models.config import MLXContextConfig

CONFIG = MLXContextConfig(
    # Model
    model_name="mlx-community/Qwen3.5-9B-MLX-4bit",
    seed=42,
    # Context sweep
    context_sizes=[512, 1024, 2048, 4096, 8192, 16384, 32768],
    # Tasks — must be keys registered in TASK_REGISTRY
    active_tasks=["bug_detection", "classifier", "dbt_model"],
    # Generation
    max_tokens=256,
    # Run control
    warmup_runs=1,
    num_runs=3,
    # Memory safety
    memory_safety_threshold=0.85,  # fraction of available RAM
    force_run=False,  # True → interactive override prompt
    generation_headroom_gb=1.0,  # buffer on top of weight + KV estimate
    # Output
    output_dir="output",
)


def emit_rich_table(results_data: list[Any]) -> None:
    """
    Print a formatted table of the results.

    Args:
        results_data: List of ContextScaleResult objects.
    """
    console = Console()
    table = Table(title="Context Scaling Benchmark Results")

    table.add_column("Task", style="cyan")
    table.add_column("Context Tokens", justify="right")
    table.add_column("TTFT (ms)", justify="right")
    table.add_column("Tok. Time (ms)", justify="right")
    table.add_column("Prefill TPS", justify="right")
    table.add_column("Decode TPS", justify="right")
    table.add_column("Mem (GB)", justify="right")
    table.add_column("Status", justify="center")

    for res in results_data:
        if res.skipped:
            table.add_row(
                res.task_id,
                str(res.target_context_tokens),
                "—",
                "—",
                "—",
                "—",
                "—",
                "[yellow]SKIP[/yellow]",
            )
        else:
            table.add_row(
                res.task_id,
                str(res.actual_context_tokens),
                f"{res.ttft_ms:.1f}",
                f"{res.tokenizer_time_ms:.1f}",
                f"{res.prompt_tps:.1f}",
                f"{res.generation_tps:.1f}",
                f"{res.peak_memory_gb:.2f}",
                "[green]✓[/green]",
            )

    console.print(table)


def main() -> None:
    """Main entrypoint for the context scaling benchmark."""
    # Validate tasks
    tasks_to_run = []
    for t_id in CONFIG.active_tasks:
        if t_id not in TASK_REGISTRY:
            raise ValueError(f"Task '{t_id}' not found in TASK_REGISTRY.")
        tasks_to_run.append(TASK_REGISTRY[t_id])

    os.makedirs(CONFIG.output_dir, exist_ok=True)

    sweep_result = run_ctx_sweep(CONFIG, tasks_to_run)

    # Save JSON
    model_slug = CONFIG.model_name.split("/")[-1]
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(
        CONFIG.output_dir, f"ctx_scale_{model_slug}_{timestamp}.json"
    )

    with open(out_path, "wb") as f:
        f.write(msgspec.json.encode(sweep_result))

    print(f"\nResults saved to {out_path}\n")

    # Emit table
    emit_rich_table(sweep_result.results)


if __name__ == "__main__":
    main()
