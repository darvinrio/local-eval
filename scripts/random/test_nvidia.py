"""
test_nvidia.py

test nvidia api keys
i mean, it works but the free models ttft is too long
"""

import os
import sys

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

_USE_COLOR = sys.stdout.isatty() and os.getenv("NO_COLOR") is None
_REASONING_COLOR = "\033[90m" if _USE_COLOR else ""
_RESET_COLOR = "\033[0m" if _USE_COLOR else ""

# NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
ZHYPRA_API_KEY = os.getenv("ZHYPRA_API_KEY")

# client = OpenAI(base_url="https://integrate.api.nvidia.com/v1", api_key=NVIDIA_API_KEY)
client = OpenAI(base_url="https://api.zyphracloud.com/api/v1", api_key=ZHYPRA_API_KEY)


completion = client.chat.completions.create(
    # model="z-ai/glm4.7",
    model="zyphra/ZAYA1-8B",
    messages=[
        {"role": "system", "content": "only reply in english"},
        {"role": "user", "content": "Explain what a dbt model is in one paragraph."},
    ],
    temperature=1,
    top_p=1,
    max_tokens=16384,
    extra_body={
        "chat_template_kwargs": {"enable_thinking": True, "clear_thinking": False}
    },
    stream=True,
)

for chunk in completion:
    if not getattr(chunk, "choices", None):
        continue
    if len(chunk.choices) == 0 or getattr(chunk.choices[0], "delta", None) is None:
        continue
    delta = chunk.choices[0].delta
    reasoning = getattr(delta, "reasoning_content", None)
    if reasoning:
        print(f"{_REASONING_COLOR}{reasoning}{_RESET_COLOR}", end="")
    if getattr(delta, "content", None) is not None:
        print(delta.content, end="")
