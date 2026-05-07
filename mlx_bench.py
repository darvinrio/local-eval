"""
mlx_bench.py — Unified MLX-LM benchmark + eval script

Runs: throughput (tok/sec, memory), perplexity, and lm-eval accuracy tasks
for one or more models.
"""
import json
from benchmarks.mlx_bm import run_mlx_bm
from benchmarks.mlx_eval import run_mlx_eval


# if __name__ == "__main__":
#     MODEL_NAME = "mlx-community/Qwen3.5-9B-MLX-4bit"
#     PROMPT = "Explain what a dbt model is in one paragraph."
#     run_mlx_bm(MODEL_NAME, PROMPT)


if __name__ == "__main__":
    MODEL_NAME = "mlx-community/Qwen3.5-9B-MLX-4bit"
    PROMPT = "Explain what a dbt model is in one paragraph."
    eval_results = run_mlx_eval(MODEL_NAME)
    with open(f"EVAL_{MODEL_NAME}.json", "w") as f:
        json.dump(eval_results, f, indent=4)
