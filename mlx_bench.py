"""
mlx_bench.py — Unified MLX-LM benchmark + eval script

Runs: throughput (tok/sec, memory), perplexity, and lm-eval accuracy tasks
for one or more models.
"""
from benchmarks.mlx_bm import run_mlx_bm


if __name__ == "__main__":
    MODEL_NAME = "mlx-community/Qwen3.5-9B-MLX-4bit"
    PROMPT = "Explain what a dbt model is in one paragraph."
    run_mlx_bm(MODEL_NAME, PROMPT)
