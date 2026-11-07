"""Orchestration dashboard data and controls.

The dashboard itself is served by llm-gateway on the Pi (http://127.0.0.1:8080/dashboard; llm-gateway
docs/DASHBOARD.md). This package keeps the orchestration part: `summarize()` (workers, batches, task
lanes, alerts) and `Controls` (start, resume and kill plans, batch outputs, task-file cleanup).
"""

from .controls import Controls
from .data import summarize

__version__ = "2.0.0"
__all__ = ["Controls", "summarize", "__version__"]
