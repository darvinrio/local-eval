"""
models/config.py

Config Variables
"""

import msgspec


class Config(msgspec.Struct):
    """Config storing variables for benchmarking"""

    LATENCY_PROMPT: str
