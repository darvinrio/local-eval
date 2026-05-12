"""
benchmarks/mlx_bm.py

MLX benchmark runner.
"""

from typing import TYPE_CHECKING, Any, cast

import mlx.core as mx
import mlx.nn as nn
from loguru import logger
from mlx_lm import load, stream_generate

from models.results import BenchmarkResult
from utils.basic import unload

if TYPE_CHECKING:
    from mlx_lm.tokenizer_utils import TokenizerWrapper


def run_mlx_bm(
    model_name: str,
    prompt: str,
    max_tokens: int = 1024,
    temp: float = 0.6,
    seed: int = 42,
) -> BenchmarkResult:
    """
    Run MLX benchmark.

    Args:
        model_name: Model name to run.
        prompt: Prompt to use.
        max_tokens: Maximum number of tokens to generate.
        temp: Temperature for sampling.
        seed: Random seed.

    Returns:
        BenchmarkResult: Result of the benchmark.
    """
    mx.random.seed(seed)

    load_result = cast(
        "tuple[nn.Module, TokenizerWrapper, dict[str, Any]]",  # workaround for typing
        load(model_name, return_config=True),
    )
    model, tokenizer, config = load_result

    messages = [{"role": "user", "content": prompt}]
    tokenized_prompt = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )

    logger.debug(tokenized_prompt)
    logger.success("")

    for response in stream_generate(
        model, tokenizer=tokenizer, prompt=tokenized_prompt, max_tokens=max_tokens
    ):
        print(response.text, end="", flush=True)

    logger.success("")
    logger.debug(response)

    bm_result = BenchmarkResult(
        prompt_tps=response.prompt_tps,
        generation_tps=response.generation_tps,
        peak_memory_gb=response.peak_memory,
        prompt_tokens=response.prompt_tokens,
        generation_tokens=response.generation_tokens,
    )
    unload(model=model, tokenizer=tokenizer)
    return bm_result


if __name__ == "__main__":
    MODEL_NAME = "mlx-community/Qwen3.5-9B-MLX-4bit"
    PROMPT = "Explain what a dbt model is in one paragraph."
    run_mlx_bm(MODEL_NAME, PROMPT)
