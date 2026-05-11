"""
benchmarks/tasks/classifier.py

Rule-based classifier task.
"""

from .base import ContextTask

# ruff: disable[E501] line too long
CLASSIFIER_TASK = ContextTask(
    task_id="classifier",
    task_name="Rule-based Classifier",
    description="Rule-based text classifier",
    system_prompt="You are a strict rule-based text classifier.",
    user_prompt_prefix="Here is a list of classification rules followed by a document to classify:\n\n",
    user_prompt_suffix="\n\nApply the rules above in order. Output only the matching rule IDs and a one-sentence justification.",
    seed_content="""[RULE-001]
Condition: Text contains the word "urgent" or "asap" (case-insensitive).
Action: Classify as URGENT.

[RULE-002]
Condition: Text mentions a specific date or time.
Action: Classify as SCHEDULED.

[RULE-003]
Condition: Text contains a question mark.
Action: Classify as INQUIRY.

Document:
Hi team,
We need to discuss the upcoming feature launch. Can we schedule a meeting for next Tuesday?
It's not urgent, but I'd like to get it on the calendar.
Thanks!
""",
    seed_content_lang="text",
)
