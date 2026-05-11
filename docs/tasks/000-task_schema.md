# TASK_SCHEMA.md — Context-Scale Benchmark Task Authoring Guide

> Copy-paste this document as a prompt to any LLM to produce a new task.
> The LLM should output a single Python file that can be dropped into
> `benchmarks/tasks/` and registered in `TASK_REGISTRY`.

---

## What a Task Is

A `ContextTask` is a realistic long-form LLM prompt whose **content section**
can be repeated to fill any target number of tokens. The model is asked to do
something useful with the content — but we are measuring *speed*, not
*correctness*.

The task must produce a prompt that:
1. Grows naturally when the content section is repeated
2. Makes semantic sense even when repeated (e.g. "here are many more functions
   to review" or "here are many more rules to apply")
3. Has a short, specific answer instruction (suffix) so `max_tokens=256`
   is sufficient for a real response

---

## Field Specification

| Field | Type | Constraints |
|---|---|---|
| `task_id` | `str` | snake_case, unique, no spaces. e.g. `"dbt_model"` |
| `task_name` | `str` | Human-readable title. e.g. `"dbt Model Generator"` |
| `description` | `str` | 1–2 sentences. What capability this tests. |
| `system_prompt` | `str` | System role. 1 sentence. No task instructions here. |
| `user_prompt_prefix` | `str` | Instruction placed **before** the context block. Should set up what the model will see. End without a newline. |
| `user_prompt_suffix` | `str` | Instruction placed **after** the context block. Should constrain the output format. May be `""`. |
| `seed_content` | `str` | The raw block that gets repeated to fill context. See guidelines below. |
| `seed_content_lang` | `str` | One of: `"python"`, `"sql"`, `"text"`, `"html"`, `"json"`, `"yaml"` |

---

## Seed Content Guidelines

- **Target length**: 250–350 tokens when tokenised with a standard BPE tokeniser.
  This gives fine-grained control over the context ladder.
- **Self-contained**: The seed must make sense when read in isolation AND when
  concatenated with itself. Use language that naturally accumulates:
  - Code tasks: multiple function definitions, classes, or modules
  - Rule tasks: additional numbered rules
  - Data tasks: additional entity/table descriptions
- **No degenerate repetition within the seed**: Don't use `"foo " * 1000`.
  Use realistic, varied tokens.
- **Language-appropriate**: Python seeds should be valid Python. SQL seeds
  should be valid SQL. Text seeds should be coherent prose.
- **Avoid markdown code fences** inside seed content — the prompt prefix
  can provide those if needed.

---

## Output Format

Produce a single Python file named `{task_id}.py` with this structure:

```python
"""
benchmarks/tasks/{task_id}.py

{task_name} context-scale benchmark task.
"""

from benchmarks.tasks.base import ContextTask

{TASK_ID_UPPER}_TASK = ContextTask(
    task_id="{task_id}",
    task_name="{task_name}",
    description=(
        "{description}"
    ),
    system_prompt=(
        "{system_prompt}"
    ),
    user_prompt_prefix=(
        "{user_prompt_prefix}"
    ),
    user_prompt_suffix=(
        "{user_prompt_suffix}"
    ),
    seed_content=(
        """\
{seed_content}
"""
    ),
    seed_content_lang="{seed_content_lang}",
)
```

Then add one line to `benchmarks/tasks/__init__.py`:
```python
from benchmarks.tasks.{task_id} import {TASK_ID_UPPER}_TASK
TASK_REGISTRY["{task_id}"] = {TASK_ID_UPPER}_TASK
```

---

## Worked Example: `bug_detection`

```python
"""
benchmarks/tasks/bug_detection.py

Bug Detection context-scale benchmark task.
"""

from benchmarks.tasks.base import ContextTask

BUG_DETECTION_TASK = ContextTask(
    task_id="bug_detection",
    task_name="Bug Detection",
    description=(
        "Tests the model's ability to scan Python source code for bugs, "
        "anti-patterns, and code smells across increasing context lengths."
    ),
    system_prompt=(
        "You are a senior Python engineer performing a thorough code review."
    ),
    user_prompt_prefix=(
        "Here is a Python source file. "
        "Identify every bug, anti-pattern, and code smell you find:"
    ),
    user_prompt_suffix=(
        "List each issue with: approximate line reference, issue type, "
        "and a one-line explanation. Be exhaustive."
    ),
    seed_content=(
        """\
# module: data_pipeline.py

def fetch_records(db, filters=[]):
    results = []
    for f in filters:
        results = db.query(f)  # overwrites results each iteration
    return results

def compute_average(values):
    total = 0
    for v in values:
        total = total + v
    return total / len(values)  # ZeroDivisionError if values is empty

class DataProcessor:
    cache = {}  # mutable class-level default — shared across instances

    def process(self, record):
        id = record["id"]  # shadows built-in 'id'
        if self.cache.get(id) != None:  # use 'is not None'
            return self.cache[id]
        result = self._transform(record)
        self.cache[id] = result

    def _transform(self, record):
        return {k: str(v) for k, v in record.items()}

def load_config(path):
    import json  # import inside function — should be at module level
    with open(path) as f:
        config = json.load(f)
    return config
"""
    ),
    seed_content_lang="python",
)
```

---

## Checklist Before Submitting a New Task

- [ ] `task_id` is snake_case and unique in `TASK_REGISTRY`
- [ ] `seed_content` is 250–350 tokens (estimate: ~4 chars/token)
- [ ] Seed reads naturally when concatenated with itself ×10
- [ ] `user_prompt_suffix` constrains the answer to fit in 256 tokens
- [ ] File is importable with no external dependencies beyond `benchmarks.tasks.base`
- [ ] Registered in `benchmarks/tasks/__init__.py`
