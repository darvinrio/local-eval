import os
import time

import tiktoken
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


api_key = os.environ.get("UNSLOTH_LOCAL_KEY")
client = OpenAI(base_url="http://localhost:8888/v1", api_key=api_key)
# MODEL_NAME = "unsloth/gemma-4-E4B-it-GGUF"
# MODEL_NAME = "Qwen3-Coder-30B-A3B-Instruct-IQ4_XS"
MODEL_NAME = "Jiunsong/supergemma4-26b-uncensored-mlx-4bit-v2"


def measure_latency(prompt: str, repetitions: int = 3, max_tokens: int = 8192):
    enc = tiktoken.get_encoding("cl100k_base")  # approximate tokenizer
    prompt_tokens = len(enc.encode(prompt))

    results = []
    for i in range(repetitions):
        t0 = time.perf_counter()

        # streaming to measure TTFT precisely
        stream = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            stream=True,
        )

        first_token_time = None
        tokens = 0

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
            tokens += len(enc.encode(delta))

        t_end = time.perf_counter()

        ttft = (first_token_time - t0) * 1000  # ms
        gen_time = t_end - first_token_time
        tok_per_sec = tokens / gen_time if gen_time > 0 else 0

        results.append(
            {
                "ttft_ms": ttft,
                "gen_toks": tokens,
                "gen_time_s": gen_time,
                "tok_per_sec": tok_per_sec,
            }
        )

    # Simple aggregate
    agg = {
        "prompt_tokens": prompt_tokens,
        "runs": len(results),
        "avg_ttft_ms": sum(r["ttft_ms"] for r in results) / len(results),
        "avg_tok_per_sec": sum(r["tok_per_sec"] for r in results) / len(results),
        "detail": results,
    }
    return agg


if __name__ == "__main__":
    prompt = "Explain what a dbt model is in one paragraph."
    metrics = measure_latency(prompt, repetitions=10)
    print(metrics)
