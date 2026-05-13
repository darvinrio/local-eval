"""
basic.py - basic utilities

unload model from memory
"""

import gc

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.tokenizer_utils import TokenizerWrapper


def unload(model: nn.Module, tokenizer: TokenizerWrapper) -> None:
    """Free model memory before loading the next one."""
    del model, tokenizer
    gc.collect()
    mx.clear_cache()
