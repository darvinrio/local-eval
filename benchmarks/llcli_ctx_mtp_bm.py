#!/usr/bin/env python3
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

# ==============================================================================
# CONFIGURATION PARAMETERS
# ==============================================================================
# Path to your llama.cpp binaries
LLAMA_CLI_PATH: str = "./llama-cli"

OUTPUT_DIR: str = "./bench_results"

# Model paths
TARGET_MODEL: str = os.path.expanduser("~/models/Qwen3.6-27B-Q4_K_S.gguf")
MTP_DRAFT_MODEL: str = os.path.expanduser(
    "~/models/Qwen3.6-MTP-Draft.gguf"
)  # Update with actual path

# Target architecture/hardware configs
GPU_LAYERS: int = 99  # Offload all to Metal
THREADS: int = 8  # Performance cores on M4

# llama-bench equivalent hyperparams
BATCH_SIZE: int = 2048
UBATCH_SIZE: int = 2048
N_PREDICT: int = 1024  # -n 1024
FLASH_ATTN: int = 1  # -fa 1
CACHE_TYPE_K: str = "q4_1"
CACHE_TYPE_V: str = "q4_1"

# Context sizes to sweep over
CONTEXT_SIZES: list[int] = [4096, 8192, 16384, 32768, 65536, 131072]

# Metadata for tracking (mimicking llama-bench schema constraints)
BUILD_COMMIT: str = "unknown"
BUILD_NUMBER: str = "0"
CPU_INFO: str = "Apple M4"
GPU_INFO: str = "Apple M4 GPU"
BACKENDS: str = "Metal"
# ==============================================================================


class MTPBenchmarkRunner:
    def __init__(self) -> None:
        self.output_path = Path(OUTPUT_DIR)
        self.output_path.mkdir(parents=True, exist_ok=True)

    def _run_with_peak_memory(self, cmd: list[str]) -> tuple[str, int]:
        """Runs command via /usr/bin/time -l to track macOS maximum resident set size."""
        time_cmd = ["/usr/bin/time", "-l"] + cmd
        print(f"\nExecuting: {' '.join(time_cmd)}")

        # Capture stderr because /usr/bin/time outputs statistics directly to stderr
        process = subprocess.Popen(
            time_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )

        stdout_content, stderr_content = process.communicate()

        # Extract peak memory (maximum resident set size) from macOS time -l
        # Format typical line: "  12845056  maximum resident set size"
        peak_rss_bytes = 0
        match = re.search(r"(\d+)\s+maximum resident set size", stderr_content)
        if match:
            peak_rss_bytes = int(match.group(1))

        # Combine output streams for metric parsing if llama targets logged to stderr
        combined_output = stdout_content + "\n" + stderr_content
        return combined_output, peak_rss_bytes

    def _parse_perf_metrics(self, output: str) -> dict[str, Any]:
        """Parses performance metrics from llama_perf_context_print string block."""
        metrics: dict[str, Any] = {
            "sample_time_ms": 0.0,
            "prompt_eval_time_ms": 0.0,
            "prompt_eval_tokens_per_second": 0.0,
            "eval_time_ms": 0.0,
            "eval_tokens_per_second": 0.0,
            "spec_eval_time_ms": 0.0,
            "spec_accept_percent": 0.0,
        }

        # Regex parsers for llama.cpp performance logs
        sample_match = re.search(r"sample time\s*=\s*([\d.]+)\s*ms", output)
        prompt_match = re.search(
            r"prompt eval time\s*=\s*([\d.]+)\s*ms\s*/\s*(\d+)\s*tokens\s*.*?\(([\d.]+)\s*t/s\)",
            output,
        )
        eval_match = re.search(
            r"eval time\s*=\s*([\d.]+)\s*ms\s*/\s*(\d+)\s*runs\s*.*?\(([\d.]+)\s*t/s\)",
            output,
        )

        # Speculative metrics variant if printed by llama-cli
        spec_match = re.search(r"spec eval time\s*=\s*([\d.]+)\s*ms", output)
        accept_match = re.search(r"accept rate\s*=\s*([\d.]+)\s*%", output)

        if sample_match:
            metrics["sample_time_ms"] = float(sample_match.group(1))
        if prompt_match:
            metrics["prompt_eval_time_ms"] = float(prompt_match.group(1))
            metrics["prompt_eval_tokens_per_second"] = float(prompt_match.group(3))
        if eval_match:
            metrics["eval_time_ms"] = float(eval_match.group(1))
            metrics["eval_tokens_per_second"] = float(eval_match.group(3))
        if spec_match:
            metrics["spec_eval_time_ms"] = float(spec_match.group(1))
        if accept_match:
            metrics["spec_accept_percent"] = float(accept_match.group(1))

        return metrics

    def build_base_schema(self, ctx_size: int, is_mtp: bool) -> dict[str, Any]:
        """Populates the standard static metadata matching llama-bench output."""
        return {
            "build_commit": BUILD_COMMIT,
            "build_number": BUILD_NUMBER,
            "cpu_info": CPU_INFO,
            "gpu_info": GPU_INFO,
            "backends": BACKENDS,
            "model_filename": os.path.basename(TARGET_MODEL),
            "model_type": "MTP" if is_mtp else "Base",
            "model_size": os.path.getsize(TARGET_MODEL)
            if os.path.exists(TARGET_MODEL)
            else 0,
            "n_batch": BATCH_SIZE,
            "n_ubatch": UBATCH_SIZE,
            "n_threads": THREADS,
            "cpu_mask": "0x0",
            "cpu_strict": 0,
            "poll": 0,
            "type_k": CACHE_TYPE_K,
            "type_v": CACHE_TYPE_V,
            "n_gpu_layers": GPU_LAYERS,
            "split_mode": "none",
            "main_gpu": 0,
            "no_kv_offload": 0,
            "flash_attn": FLASH_ATTN,
            "tensor_split": "0",
            "use_mmap": 1,
            "embeddings": 0,
            "n_prompt": ctx_size,
            "n_gen": N_PREDICT,
            "n_depth": 1,
        }

    def run_benchmark(self, ctx_size: int, use_mtp: bool) -> None:
        """Executes a single structural context run and exports metrics straight to disk."""
        mode_str = "MTP" if use_mtp else "Normal"
        print(f"\n>>> Running context size {ctx_size} in {mode_str} mode...")

        # Setup standard llama-cli execution layout
        cmd = [
            LLAMA_CLI_PATH,
            "-m",
            TARGET_MODEL,
            "-b",
            str(BATCH_SIZE),
            "-ub",
            str(UBATCH_SIZE),
            "-c",
            str(ctx_size),
            "-n",
            str(N_PREDICT),
            "-ngl",
            str(GPU_LAYERS),
            "-t",
            str(THREADS),
            "--ctk",
            CACHE_TYPE_K,
            "--ctv",
            CACHE_TYPE_V,
            # Generate dummy prompt matching targeted prompt depth context
            "-p",
            "hello " * (ctx_size // 2),
        ]

        if FLASH_ATTN:
            cmd.append("-fa")

        # Inject speculative MTP draft routing parameters if enabled
        if use_mtp:
            cmd.extend(["--draft", str(N_PREDICT), "-md", MTP_DRAFT_MODEL])

        try:
            raw_output, peak_rss = self._run_with_peak_memory(cmd)
            perf_metrics = self._parse_perf_metrics(raw_output)

            # Formulate full final structural schema payload
            result_data = self.build_base_schema(ctx_size, use_mtp)
            result_data.update(
                {
                    "peak_memory_bytes": peak_rss,
                    "peak_memory_mb": round(peak_rss / (1024 * 1024), 2),
                    "metrics": perf_metrics,
                    # Mapping processing speed to native llama-bench target keys
                    "avg_ts": perf_metrics["eval_tokens_per_second"],
                    "avg_ns": int(perf_metrics["eval_time_ms"] * 1_000_000)
                    if perf_metrics["eval_time_ms"]
                    else 0,
                }
            )

            # Save immediately to shield against following OOM crashes
            file_name = f"result_{mode_str.lower()}_ctx_{ctx_size}.json"
            target_file = self.output_path / file_name

            with open(target_file, "w") as f:
                json.dump(result_data, f, indent=2)

            print(f"✓ Saved results to {target_file}")
            print(
                f"  Tokens/sec: {perf_metrics['eval_tokens_per_second']} t/s | Peak RSS: {result_data['peak_memory_mb']} MB"
            )

        except Exception as e:
            print(
                f"✕ Benchmark failed for context {ctx_size}: {str(e)}", file=sys.stderr
            )

    def execute_suite(self) -> None:
        """Executes full sequence across normal models followed cleanly by MTP configurations."""
        # 1. Normal Model Runs
        for ctx in CONTEXT_SIZES:
            self.run_benchmark(ctx_size=ctx, use_mtp=False)

        # 2. MTP Model Runs
        for ctx in CONTEXT_SIZES:
            self.run_benchmark(ctx_size=ctx, use_mtp=True)


if __name__ == "__main__":
    runner = MTPBenchmarkRunner()
    runner.execute_suite()
