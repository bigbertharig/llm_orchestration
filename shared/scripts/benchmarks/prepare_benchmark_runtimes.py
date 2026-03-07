#!/usr/bin/env python3
"""Benchmark wrapper for canonical runtime prep script.

Canonical implementation lives at:
  /mnt/shared/scripts/prepare_llm_runtimes.py
"""

from __future__ import annotations

import os
import subprocess
import sys


def main() -> int:
    target = "/mnt/shared/scripts/prepare_llm_runtimes.py"
    cmd = ["python3", target] + sys.argv[1:]
    proc = subprocess.run(cmd, check=False, env=os.environ.copy())
    return int(proc.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
