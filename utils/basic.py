"""
basic.py - basic utilities

unload model from memory
"""

import gc

import mlx.core as mx


def unload(model, tokenizer) -> None:
    """Free model memory before loading the next one."""
    del model, tokenizer
    gc.collect()
    mx.metal.clear_cache()
