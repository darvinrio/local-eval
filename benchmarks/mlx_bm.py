"""
benchmarks/mlx_bm.py

MLX benchmark runner.
"""
from typing import cast, Any

import time
import mlx.nn as nn
import mlx.core as mx
from loguru import logger
from mlx_lm import stream_generate, load
from mlx_lm.tokenizer_utils import TokenizerWrapper

from models.results import BenchmarkResult
from utils.basic import unload

def run_mlx_bm(
    model_name: str,
    prompt: str,
    max_tokens: int = 1024,
    temp: float = 0.6,
    seed: int = 42,
) -> BenchmarkResult:
    load_result = cast(
        tuple[nn.Module, TokenizerWrapper, dict[str, Any]], # workaround for typing
        load(model_name, return_config=True),
    )
    model, tokenizer, config = load_result
    mx.random.seed(seed)

    messages = [{"role": "user", "content": prompt}]
    tokenized_prompt = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )

    logger.debug(tokenized_prompt)
    logger.success("")

    for response in stream_generate(
        model,
        tokenizer=tokenizer,
        prompt=tokenized_prompt,
        max_tokens=max_tokens
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
