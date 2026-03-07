#!/usr/bin/env python3
"""Start rig in default mode via existing return_default flow."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> int:
    target = Path(__file__).resolve().parent / "return_default.py"
    cmd = [sys.executable, str(target)] + sys.argv[1:]
    proc = subprocess.run(cmd, check=False)
    return int(proc.returncode)


if __name__ == "__main__":
    raise SystemExit(main())

