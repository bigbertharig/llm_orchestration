"""LLM Orchestration Dashboard Package.

A web dashboard for monitoring and controlling the LLM orchestration system.
Provides real-time status of workers, batches, tasks, and system health.

Usage:
    python -m dashboard --port 8787

Or import and use programmatically:
    from dashboard import main, summarize
    main()
"""

from .data import summarize
from .server import main

__version__ = "1.0.0"
__all__ = ["main", "summarize", "__version__"]
