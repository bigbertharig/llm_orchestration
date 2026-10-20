#!/usr/bin/env python3
"""Forwarder: the CPU agent lives at shared/scripts/cpu_agent.py (the single source of truth).

CPU workers run it from the shared drive (/media/bryan/shared/scripts/cpu_agent.py, see
shared/workspace/cpu-worker-setup.md). This copy only keeps `python3 scripts/cpu_agent.py ...` working
from a repo checkout; it runs the shared/ version with the same arguments.
"""

import runpy
import sys
from pathlib import Path

TARGET = Path(__file__).resolve().parents[1] / "shared" / "scripts" / "cpu_agent.py"
sys.argv[0] = str(TARGET)
runpy.run_path(str(TARGET), run_name="__main__")
