"""
benchmarks/tasks/__init__.py

This module provides access to the various benchmarks tasks used by the eval harness.
"""

from .base import ContextTask
from .bug_detection import BUG_DETECTION_TASK
from .classifier import CLASSIFIER_TASK
from .dbt_model import DBT_MODEL_TASK

TASK_REGISTRY: dict[str, ContextTask] = {
    "bug_detection": BUG_DETECTION_TASK,
    "classifier": CLASSIFIER_TASK,
    "dbt_model": DBT_MODEL_TASK,
}
