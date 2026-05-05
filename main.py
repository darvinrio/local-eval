import json
import os
import re
import time

from dotenv import load_dotenv
from openai import OpenAI
from transformers import AutoTokenizer

load_dotenv()

api_key = os.environ.get("UNSLOTH_LOCAL_KEY")
client = OpenAI(base_url="http://localhost:8888/v1", api_key=api_key)
# MODEL_NAME = "unsloth/gemma-4-E4B-it-GGUF"
# MODEL_NAME = "Qwen3-Coder-30B-A3B-Instruct-IQ4_XS"
# MODEL_NAME = "Jiunsong/supergemma4-26b-uncensored-mlx-4bit-v2"
MODEL_NAME = "leonsarmiento/Qwen3.6-27B-3bit-mlx"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


def sanitize_filename(name: str) -> str:
    basename = name.split("/")[-1]
    return re.sub(r"[^\w\-.]", "_", basename)


def measure_latency(prompt: str, repetitions: int = 3, max_tokens: int = 8192):
    prompt_tokens = len(tokenizer.encode(prompt))

    results = []
    for i in range(repetitions):
        print(f"  Run {i + 1}/{repetitions}...", end=" ", flush=True)

        t0 = time.perf_counter()

        stream = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            stream=True,
            stream_options={"include_usage": True},
        )

        first_token_time = None
        tokens = 0
        generated_text = []
        usage = None

        for chunk in stream:
            now = time.perf_counter()
            if first_token_time is None:
                first_token_time = now

            if (
                chunk.choices
                and chunk.choices[0].delta
                and chunk.choices[0].delta.content
            ):
                delta = chunk.choices[0].delta.content
            else:
                delta = ""

            if delta:
                generated_text.append(delta)
                tokens += len(tokenizer.encode(delta))

            if chunk.usage:
                usage = chunk.usage

        t_end = time.perf_counter()
        tokens = usage.completion_tokens
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
    filepath = f"output/oai-{filename}.json"
    with open(filepath, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nResults saved → {filepath}")
    return filepath


if __name__ == "__main__":
    prompt = "Explain what a dbt model is in one paragraph."
    print(f"Model : {MODEL_NAME}")
    print(f"Server: {client.base_url}")
    print(f"Prompt: {prompt!r}\n")

    metrics = measure_latency(prompt, repetitions=10)
    save_results(metrics)

    print(f"\n=== Aggregate Results ===")
    print(f"Prompt tokens   : {metrics['prompt_tokens']}")
    print(f"Avg TTFT        : {metrics['avg_ttft_ms']:.1f} ms")
    print(f"Avg throughput  : {metrics['avg_tok_per_sec']:.1f} tok/s")
    print(f"Runs            : {metrics['runs']}")
