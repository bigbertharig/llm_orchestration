#!/usr/bin/env python3
"""Start rig in empty worker mode.

This is the canonical "empty" startup entrypoint:
- brain loaded
- workers started cold
- auto-default disabled
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> int:
    target = Path(__file__).resolve().parent / "start_benchmark_mode.py"
    cmd = [sys.executable, str(target)] + sys.argv[1:]
    proc = subprocess.run(cmd, check=False)
    return int(proc.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
