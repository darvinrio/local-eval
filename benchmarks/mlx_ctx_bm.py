"""
benchmarks/mlx_ctx_bm.py

Context scaling benchmark runner.
"""

import gc
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
from utils.kv_cache import estimate_kv_cache


def _build_context(
    task: ContextTask, target_tokens: int, tokenizer: TokenizerWrapper
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


def _resolve_text_config(config: dict[str, Any]) -> dict[str, Any]:
    """
    Extract the text model config from a potentially nested model config.

    Multimodal models (Gemma4ForConditionalGeneration,
    Qwen3_5MoeForConditionalGeneration, etc.) nest architecture params
    under 'text_config' or 'language_config'. Text-only models have them
    at the root level.

    Args:
        config: Raw config dict from mlx_lm.load(..., return_config=True).

    Returns:
        The sub-dict containing num_hidden_layers, num_key_value_heads, etc.
    """
    for sub_key in ("text_config", "language_config", "llm_config"):
        if sub_key in config and isinstance(config[sub_key], dict):
            sub = config[sub_key]
            if "num_hidden_layers" in sub or "num_layers" in sub:
                return sub
    return config


def _memory_preflight(
    task: ContextTask,
    context_tokens: int,
    config: dict[str, Any],
    force_run: bool,
    safety_threshold: float,
    generation_headroom_gb: float,
    pre_load_available_gb: float,
    kv_quant_bits: int = 16,
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
        kv_quant_bits: Bits per KV element. 16 = bf16 (default),
                       8 or 4 for quantised KV cache testing.

    Returns:
        tuple[bool, str]: Whether memory preflight passed and reason.
    """
    estimate = estimate_kv_cache(config, kv_quant_bits=kv_quant_bits)
    kv_cache_gb = estimate.estimate_gb(context_tokens)

    model_weights_gb = mx.get_active_memory() / 1e9
    required_gb = model_weights_gb + kv_cache_gb + generation_headroom_gb

    logger.debug(
        f"Preflight: {estimate.model_type} kv_bits={estimate.kv_quant_bits} "
        f"kv_cache={kv_cache_gb:.2f} GiB | breakdown: "
        + ", ".join(
            f"{lb.layer_type}({lb.num_layers}L)={lb.bytes_per_token_per_layer}B/tok/L"
            + (f" cap={lb.max_cache_tokens}" if lb.max_cache_tokens else "")
            for lb in estimate.layer_breakdowns
        )
    )

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
                f"⚠  {context_tokens} tokens may use {required_gb:.1f} GB "
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
    run_index: int = 0,
    config: MLXContextConfig | None = None,
) -> ContextScaleRunResult:
    """
    Run a single inference run.

    Args:
        model: Model to run inference on.
        tokenizer: Tokenizer to use.
        messages: Messages to send to the model.
        max_tokens: Maximum number of tokens to generate.
        run_index: Index of the run.
        config: Configuration for the benchmark.

    Returns:
        ContextScaleRunResult: Metrics for the run.
    """
    t0 = time.perf_counter()
    tokenized_prompt = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True
    )
    prompt_tokens = len(tokenized_prompt)
    tokenizer_time_ms = (time.perf_counter() - t0) * 1000

    gen = stream_generate(
        model, tokenizer=tokenizer, prompt=tokenized_prompt, max_tokens=max_tokens
    )

    t_start = time.perf_counter()
    prev_time = t_start
    t_first_token = None

    per_token_timings: list[TokenTiming] = []
    response = None
    t_last_recorded: float = t_start  # timestamp of the most recently stored token

    # Config options
    capture_traces = config.capture_per_token_timings if config else True
    max_trace_tokens = config.per_token_timing_max_tokens if config else None
    stride = config.trace_stride if config else 1
    include_final = config.include_final_token_in_trace if config else True

    for i, response in enumerate(gen):
        now = time.perf_counter()
        token_index = i + 1  # 1-based index

        if t_first_token is None:
            t_first_token = now

        if capture_traces:
            # Check if we should record this token
            should_record = False
            if max_trace_tokens is None or token_index <= max_trace_tokens:
                if (token_index - 1) % stride == 0:
                    should_record = True

            if should_record:
                delta_ms = (now - prev_time) * 1000
                since_start_ms = (now - t_start) * 1000
                since_first_token_ms = (now - t_first_token) * 1000

                per_token_timings.append(
                    TokenTiming(
                        token_index=token_index,
                        delta_ms=delta_ms,
                        since_start_ms=since_start_ms,
                        since_first_token_ms=since_first_token_ms,
                        sequence_length=prompt_tokens + token_index,
                    )
                )
                t_last_recorded = now

        prev_time = now

    if response is None or t_first_token is None:
        raise RuntimeError("No tokens generated")

    # If the final generated token was skipped by stride, append it explicitly.
    # delta_ms is measured from t_last_recorded (the previous stored entry),
    # not from prev_time which always equals now after the loop.
    final_token_index = response.generation_tokens
    if (
        capture_traces
        and include_final
        and per_token_timings
        and (max_trace_tokens is None or final_token_index <= max_trace_tokens)
        and per_token_timings[-1].token_index != final_token_index
    ):
        per_token_timings.append(
            TokenTiming(
                token_index=final_token_index,
                delta_ms=(now - t_last_recorded) * 1000,
                since_start_ms=(now - t_start) * 1000,
                since_first_token_ms=(now - t_first_token) * 1000,
                sequence_length=prompt_tokens + final_token_index,
            )
        )

    ttft_ms = (t_first_token - t_start) * 1000

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
    config: MLXContextConfig, tasks: list[ContextTask]
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

    text_config = _resolve_text_config(model_config)

    results: list[ContextScaleResult] = []

    try:
        for task in tasks:
            consecutive_errors = 0
            for context_size in config.context_sizes:
                if consecutive_errors >= 3:
                    logger.warning(
                        f"Task {task.task_id} hit 3 consecutive errors, "
                        f"bailing out of remaining context sizes."
                    )
                    break

                actual_tokens = 0
                try:
                    logger.info(f"Preparing task {task.task_id} at size {context_size}")

                    messages, actual_tokens = _build_context(
                        task, context_size, tokenizer
                    )

                    safe, reason = _memory_preflight(
                        task,
                        actual_tokens,
                        text_config,
                        config.force_run,
                        config.memory_safety_threshold,
                        config.generation_headroom_gb,
                        pre_load_available_gb,
                        kv_quant_bits=config.kv_quant_bits,
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

                    # Warmup
                    for _ in range(config.warmup_runs):
                        _single_run(
                            model,
                            tokenizer,
                            messages,
                            config.max_tokens,
                            config=None,
                        )

                    # Measure
                    measured_runs: list[ContextScaleRunResult] = []
                    num_runs = config.num_runs
                    for i in range(num_runs):
                        measured_runs.append(
                            _single_run(
                                model,
                                tokenizer,
                                messages,
                                config.max_tokens,
                                run_index=i + 1,
                                config=config,
                            )
                        )

                    # Aggregate
                    avg_tok_time = (
                        sum(r.tokenizer_time_ms for r in measured_runs) / num_runs
                    )
                    avg_ttft = sum(r.ttft_ms for r in measured_runs) / num_runs
                    avg_ptps = sum(r.prompt_tps for r in measured_runs) / num_runs
                    avg_gtps = sum(r.generation_tps for r in measured_runs) / num_runs
                    avg_peak_mem = (
                        sum(r.peak_memory_gb for r in measured_runs) / num_runs
                    )
                    # Use generation tokens from the last run
                    gen_tokens = measured_runs[-1].generation_tokens

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
                        runs=measured_runs,
                    )
                    results.append(res)

                    logger.success(
                        f"Task {task.task_id} @ {context_size} done. "
                        f"TTFT: {avg_ttft:.1f}ms"
                    )
                    consecutive_errors = 0  # reset on success

                except Exception as exc:
                    logger.error(f"Task {task.task_id} @ {context_size} FAILED: {exc}")
                    results.append(
                        ContextScaleResult(
                            task_id=task.task_id,
                            target_context_tokens=context_size,
                            actual_context_tokens=actual_tokens,
                            tokenizer_time_ms=0,
                            ttft_ms=0,
                            prompt_tps=0,
                            generation_tps=0,
                            peak_memory_gb=0,
                            generation_tokens=0,
                            skipped=True,
                            skip_reason=f"error: {exc}",
                        )
                    )
                    # Clear GPU caches to attempt recovery for subsequent sizes
                    mx.clear_cache()
                    gc.collect()
                    consecutive_errors += 1
                    continue
    finally:
        unload(model=model, tokenizer=tokenizer)

    return ContextScaleSweepResult(
        model=model_name,
        run_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        config=config,
        results=results,
    )
