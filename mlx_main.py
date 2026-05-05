import json
import os
import re
import time

import mlx.core as mx
from mlx_lm import load, stream_generate

MODEL_NAME = "Jiunsong/supergemma4-26b-uncensored-mlx-4bit-v2"

model, tokenizer = load(MODEL_NAME)


def sanitize_filename(name: str) -> str:
    """Convert a model name/path to a safe filename."""
    # Take only the last part after '/' (repo name), then replace unsafe chars
    basename = name.split("/")[-1]
    return re.sub(r"[^\w\-.]", "_", basename)


def measure_latency(prompt: str, repetitions: int = 3, max_tokens: int = 8192):
    messages = [{"role": "user", "content": prompt}]
    formatted_prompt = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )
    prompt_tokens = len(tokenizer.encode(formatted_prompt))

    results = []
    for i in range(repetitions):
        print(f"  Run {i + 1}/{repetitions}...", end=" ", flush=True)

        t0 = time.perf_counter()
        first_token_time = None
        tokens = 0
        generated_text = []

        for response in stream_generate(
            model,
            tokenizer,
            formatted_prompt,
            max_tokens=max_tokens,
        ):
            now = time.perf_counter()
            if first_token_time is None:
                first_token_time = now

            chunk_text = response.text
            if chunk_text:
                generated_text.append(chunk_text)
                tokens += len(tokenizer.encode(chunk_text, add_special_tokens=False))

        t_end = time.perf_counter()
        mx.eval()

        ttft = (first_token_time - t0) * 1000
        gen_time = t_end - first_token_time
        tok_per_sec = tokens / gen_time if gen_time > 0 else 0

        print(f"TTFT={ttft:.0f}ms  {tok_per_sec:.1f} tok/s  ({tokens} tokens)")

        results.append(
            {
                "ttft_ms": ttft,
                "gen_toks": tokens,
                "gen_time_s": gen_time,
                "tok_per_sec": tok_per_sec,
                "output_preview": "".join(generated_text)[:120],
            }
        )

    agg = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "prompt_tokens": prompt_tokens,
        "runs": len(results),
        "avg_ttft_ms": sum(r["ttft_ms"] for r in results) / len(results),
        "avg_tok_per_sec": sum(r["tok_per_sec"] for r in results) / len(results),
        "detail": results,
    }
    return agg


def save_results(metrics: dict):
    os.makedirs("output", exist_ok=True)
    filename = sanitize_filename(metrics["model"])
    filepath = f"output/mlx-{filename}.json"
    with open(filepath, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nResults saved → {filepath}")
    return filepath


if __name__ == "__main__":
    prompt = "Explain what a dbt model is in one paragraph."
    print(f"Model: {MODEL_NAME}")
    print(f"Prompt: {prompt!r}\n")

    metrics = measure_latency(prompt, repetitions=10)
    save_results(metrics)

    print(f"\n=== Aggregate Results ===")
    print(f"Prompt tokens   : {metrics['prompt_tokens']}")
    print(f"Avg TTFT        : {metrics['avg_ttft_ms']:.1f} ms")
    print(f"Avg throughput  : {metrics['avg_tok_per_sec']:.1f} tok/s")
    print(f"Runs            : {metrics['runs']}")
