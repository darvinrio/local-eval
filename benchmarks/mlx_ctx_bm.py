"""
benchmarks/mlx_ctx_bm.py

Context scaling benchmark runner.
"""

import time
from typing import Any, cast

import mlx.core as mx
import mlx.nn as nn
import psutil
from loguru import logger
from mlx_lm import load, stream_generate
from mlx_lm.tokenizer_utils import TokenizerWrapper

from benchmarks.tasks.base import ContextTask
from models.config import MLXContextConfig
from models.results import (
    ContextScaleResult,
    ContextScaleRunResult,
    ContextScaleSweepResult,
    TokenTiming,
)
from utils.basic import unload


def _build_context(
    task: ContextTask,
    target_tokens: int,
    tokenizer: TokenizerWrapper,
) -> tuple[list[dict[str, str]], int]:
    """
    Build context for the task.

    Args:
        task: Task to build context for.
        target_tokens: Target number of tokens.
        tokenizer: Tokenizer to use.

    Returns:
        tuple[list[dict[str, str]], int]: Context and number of tokens.
    """
    if not task.seed_content:
        raise ValueError("task.seed_content must not be empty")

    multiplier = (target_tokens * 10) // len(task.seed_content) + 10
    raw_blob = task.seed_content * multiplier

    blob_tokens = tokenizer.encode(raw_blob)
    sliced_tokens = blob_tokens[:target_tokens]
    scaled_context = tokenizer.decode(sliced_tokens)

    messages = [
        {"role": "system", "content": task.system_prompt},
        {
            "role": "user",
            "content": f"{task.user_prompt_prefix}\n{scaled_context}\n{task.user_prompt_suffix}",  # noqa: E501
        },
    ]

    final_tokens = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True
    )
    return messages, len(final_tokens)


def _memory_preflight(
    task: ContextTask,
    context_tokens: int,
    config: dict[str, Any],
    force_run: bool,
    safety_threshold: float,
    generation_headroom_gb: float,
    pre_load_available_gb: float,
) -> tuple[bool, str]:
    """
    Preflight memory usage for the task.

    Args:
        task: Task to preflight memory for.
        context_tokens: Number of tokens.
        config: Model configuration.
        force_run: Force run despite memory warnings.
        safety_threshold: Safety threshold for memory usage.
        generation_headroom_gb: Headroom for generation.
        pre_load_available_gb: Available system memory captured *before* model
            load.

    Returns:
        tuple[bool, str]: Whether memory preflight passed and reason.
    """
    num_layers = config.get("num_hidden_layers", config.get("num_layers", 0))
    num_kv_heads = config.get("num_key_value_heads", config.get("num_kv_heads", 0))
    head_dim = config.get(
        "head_dim", config.get("hidden_size", 0) // config.get("num_attention_heads", 1)
    )
    if head_dim == 0 and "hidden_size" in config and "num_attention_heads" in config:
        head_dim = config["hidden_size"] // config["num_attention_heads"]

    dtype_bytes = 2

    kv_cache_gb = (
        context_tokens * num_layers * num_kv_heads * head_dim * 2 * dtype_bytes
    ) / 1e9

    model_weights_gb = mx.metal.get_active_memory() / 1e9
    required_gb = model_weights_gb + kv_cache_gb + generation_headroom_gb

    if required_gb > pre_load_available_gb * safety_threshold:
        reason = (
            f"Required memory ({required_gb:.1f} GB) exceeds threshold of "
            f"pre-load available ({pre_load_available_gb:.1f} GB)"
        )
        if not force_run:
            logger.warning(f"Skipping {task.task_id} at {context_tokens}: {reason}")
            return False, reason
        else:
            ans = input(
                f"\u26a0  {context_tokens} tokens may use {required_gb:.1f} GB "
                f"(pre-load available: {pre_load_available_gb:.1f} GB). Proceed? [y/N]:"
            )
            if ans.lower() != "y":
                return False, reason

    return True, ""


def _single_run(
    model: nn.Module,
    tokenizer: TokenizerWrapper,
    messages: list[dict[str, str]],
    max_tokens: int,
    prompt_tokens: int,
    config: MLXContextConfig,
    run_index: int,
) -> ContextScaleRunResult:
    """
    Run a single inference run and collect per-token timings.

    Args:
        model: Model to run inference on.
        tokenizer: Tokenizer to use.
        messages: Messages to send to the model.
        max_tokens: Maximum number of tokens to generate.
        prompt_tokens: Number of tokens in the pre-filled prompt.
        config: Benchmark configuration (controls trace capture).
        run_index: 1-based index of this measured run.

    Returns:
        ContextScaleRunResult: Structured result with aggregate and per-token data.
    """
    # --- Tokenize ---
    t0 = time.perf_counter()
    tokenized_prompt = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True
    )
    tokenizer_time_ms = (time.perf_counter() - t0) * 1000

    # --- Generation ---
    gen = stream_generate(
        model, tokenizer=tokenizer, prompt=tokenized_prompt, max_tokens=max_tokens
    )

    t_start = time.perf_counter()
    t_first_token: float | None = None
    prev_time = t_start  # used for delta_ms

    # Trace capture control
    capture = config.capture_per_token_timings
    max_stored = config.per_token_timing_max_tokens or max_tokens
    stride = config.trace_stride
    include_final = config.include_final_token_in_trace

    per_token_timings: list[TokenTiming] = []
    stored_count = 0

    response = None
    for response in gen:
        now = time.perf_counter()

        if t_first_token is None:
            t_first_token = now
            since_first_ms = 0.0
        else:
            since_first_ms = (now - t_first_token) * 1000.0

        token_index = response.generation_tokens  # 1-based count of generated tokens
        delta_ms = (now - prev_time) * 1000.0
        since_start_ms = (now - t_start) * 1000.0
        seq_len = prompt_tokens + token_index

        prev_time = now

        # Decide whether to store this token's timing
        if capture and stored_count < max_stored:
            # stride logic: store tokens whose (token_index - 1) % stride == 0
            should_store = ((token_index - 1) % stride == 0) or (
                include_final and token_index == max_tokens
            )
            # Always include final token if we haven't stored it yet
            if should_store or token_index == max_tokens:
                token_timing = TokenTiming(
                    token_index=token_index,
                    delta_ms=delta_ms,
                    since_start_ms=since_start_ms,
                    since_first_token_ms=since_first_ms,
                    sequence_length=seq_len,
                )
                per_token_timings.append(token_timing)
                stored_count += 1

    if response is None or t_first_token is None:
        raise RuntimeError("No tokens generated")

    ttft_ms = (t_first_token - t_start) * 1000.0

    return ContextScaleRunResult(
        run_index=run_index,
        tokenizer_time_ms=tokenizer_time_ms,
        ttft_ms=ttft_ms,
        prompt_tps=response.prompt_tps,
        generation_tps=response.generation_tps,
        peak_memory_gb=response.peak_memory,
        generation_tokens=response.generation_tokens,
        prompt_tokens=prompt_tokens,
        per_token_timings=per_token_timings,
    )


def run_ctx_sweep(
    config: MLXContextConfig,
    tasks: list[ContextTask],
) -> ContextScaleSweepResult:
    """
    Run context scale benchmark.

    Args:
        config: Configuration for the benchmark.
        tasks: List of tasks to run.

    Returns:
        ContextScaleSweepResult: Results of the benchmark.
    """
    model_name = config.model_name
    mx.random.seed(config.seed)

    pre_load_available_gb = psutil.virtual_memory().available / 1e9
    logger.info(f"Pre-load available memory: {pre_load_available_gb:.2f} GB")

    logger.info(f"Loading model {model_name}...")
    load_result = cast(
        "tuple[nn.Module, TokenizerWrapper, dict[str, Any]]",
        load(model_name, return_config=True),
    )
    model, tokenizer, model_config = load_result

    results: list[ContextScaleResult] = []

    try:
        for task in tasks:
            for context_size in config.context_sizes:
                logger.info(f"Preparing task {task.task_id} at size {context_size}")

                messages, actual_tokens = _build_context(task, context_size, tokenizer)

                safe, reason = _memory_preflight(
                    task,
                    actual_tokens,
                    model_config,
                    config.force_run,
                    config.memory_safety_threshold,
                    config.generation_headroom_gb,
                    pre_load_available_gb,
                )

                if not safe:
                    results.append(
                        ContextScaleResult(
                            task_id=task.task_id,
                            target_context_tokens=context_size,
                            actual_context_tokens=0,
                            tokenizer_time_ms=0,
                            ttft_ms=0,
                            prompt_tps=0,
                            generation_tps=0,
                            peak_memory_gb=0,
                            generation_tokens=0,
                            skipped=True,
                            skip_reason=reason,
                        )
                    )
                    continue

                # Warmup (warmup runs do not store per-token traces in output)
                for _ in range(config.warmup_runs):
                    _single_run(
                        model,
                        tokenizer,
                        messages,
                        config.max_tokens,
                        actual_tokens,
                        config,
                        run_index=0,  # warmup; not stored
                    )

                # Measure
                run_results: list[ContextScaleRunResult] = []
                num_runs = config.num_runs
                for i in range(num_runs):
                    run_result = _single_run(
                        model,
                        tokenizer,
                        messages,
                        config.max_tokens,
                        actual_tokens,
                        config,
                        run_index=i + 1,
                    )
                    run_results.append(run_result)

                # Aggregate top-level averages from measured runs
                avg_tok_time = sum(r.tokenizer_time_ms for r in run_results) / num_runs
                avg_ttft = sum(r.ttft_ms for r in run_results) / num_runs
                avg_ptps = sum(r.prompt_tps for r in run_results) / num_runs
                avg_gtps = sum(r.generation_tps for r in run_results) / num_runs
                avg_peak_mem = sum(r.peak_memory_gb for r in run_results) / num_runs
                gen_tokens = run_results[-1].generation_tokens

                res = ContextScaleResult(
                    task_id=task.task_id,
                    target_context_tokens=context_size,
                    actual_context_tokens=actual_tokens,
                    tokenizer_time_ms=avg_tok_time,
                    ttft_ms=avg_ttft,
                    prompt_tps=avg_ptps,
                    generation_tps=avg_gtps,
                    peak_memory_gb=avg_peak_mem,
                    generation_tokens=gen_tokens,
                    runs=run_results,
                )
                results.append(res)

                logger.success(
                    f"Task {task.task_id} @ {context_size} done. TTFT: {avg_ttft:.1f}ms"
                )
    finally:
        unload(model=model, tokenizer=tokenizer)

    return ContextScaleSweepResult(
        model=model_name,
        run_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        config=config,
        results=results,
    )
