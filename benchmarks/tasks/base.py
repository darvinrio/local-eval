"""
benchmarks/tasks/base.py

Base class for context tasks.
"""

import dataclasses


@dataclasses.dataclass
class ContextTask:
    """Context task base class."""

    task_id: str  # snake_case unique ID, e.g. "bug_detection"
    task_name: str  # Human-readable, e.g. "Bug Detection"
    description: str  # What the task tests (used in logs and output)
    system_prompt: str  # System role message
    user_prompt_prefix: str  # Instruction placed BEFORE the scaled context
    user_prompt_suffix: str  # Instruction placed AFTER the scaled context (may be "")
    seed_content: str  # Raw text/code that gets repeated to fill context
    seed_content_lang: str  # Hint for display: "python", "sql", "text", "html"
