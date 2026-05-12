"""
scripts/random/prep_large_context.py

Simple script to test the context building.
"""

from typing import Any, cast

import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load
from mlx_lm.tokenizer_utils import TokenizerWrapper

from benchmarks.mlx_ctx_bm import _build_context
from benchmarks.tasks import TASK_REGISTRY

MODEL_NAME = "mlx-community/Qwen3.5-9B-MLX-4bit"
SEED = 42
TASK = "dbt_model"
CONTEXT_SIZE = 32768

model_name = MODEL_NAME
mx.random.seed(SEED)

load_result = cast(
    tuple[nn.Module, TokenizerWrapper, dict[str, Any]],
    load(model_name, return_config=True),
)
model, tokenizer, model_config = load_result


messages, actual_tokens = _build_context(TASK_REGISTRY[TASK], CONTEXT_SIZE, tokenizer)

print(messages, actual_tokens)
