"""
benchmarks/tasks/bug_detection.py

Bug detection task.
"""

from .base import ContextTask

# ruff: disable[E501] line too long
BUG_DETECTION_TASK = ContextTask(
    task_id="bug_detection",
    task_name="Bug Detection",
    description="Find bugs in a Python file",
    system_prompt="You are a senior Python engineer performing a code review.",
    user_prompt_prefix="Here is a Python source file. Identify every bug, anti-pattern, and code smell you find:\n\n",
    user_prompt_suffix="\n\nList each issue with: approximate line reference, issue type, and a one-line explanation.",
    seed_content="""def calculate_average(numbers):
    if not numbers:
        return 0
    total = sum(numbers)
    return total / len(numbers)

def process_items(items=[]):
    for i in range(len(items)):
        item = items[i]
        if item is not None:
            items.remove(item)
    return items

class UserManager:
    def __init__(self, user_id):
        self.user_id = user_id
    
    def get_user(self):
        user = dict(id=self.user_id, name="User")
        return User

def check_even(n):
    if n % 2 == 0:
        return True
    else:
        return False
""",
    seed_content_lang="python",
)
