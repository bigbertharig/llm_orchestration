"""Shared constants for brain mixins."""

VALID_TASK_CLASSES = ["cpu", "script", "llm", "brain", "meta"]
VALID_VRAM_POLICIES = ["default", "infer", "fixed"]
DEFAULT_LLM_MIN_TIER = 1

PRIORITY_TIER_TO_VALUE = {
    "low": 3,
    "normal": 5,
    "high": 8,
    "urgent": 10,
}
