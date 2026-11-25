"""Host prompt-cache policy: how much system RAM a runtime may keep for past prompts (DESIGN_NOTES.md).

llama-server's `--cache-ram` keeps past prompt state in system RAM so a later request sharing a
prefix skips re-processing it. System RAM is the rig's scarce resource, so the rig's floor is 0 and
this module decides per load:

- by use: workers (one-shot system prompts) and benchmarks (independent prompts) get none; chat and
  coding sessions (multi-turn reuse, returning to earlier conversations) get a bounded cache;
- by what the rig has free right now (`rig observe` guard reading), keeping a reserve so the load
  never pushes the rig toward the guard's memory lines.
"""

from __future__ import annotations

USES = ("chat", "worker", "benchmark")
CACHE_MIB_BY_USE = {"chat": 4096, "worker": 0, "benchmark": 0}
# Never plan a cache into the last part of RAM: the guard warns at 4096 MB available and acts at 2048.
RAM_RESERVE_MB = 8192
STEP_MIB = 256


def cache_ram_mib(use: str, mem_available_mb: int | None) -> int:
    """Cache for a new runtime of this use, given the rig's available RAM (None: unknown, so none)."""
    want = CACHE_MIB_BY_USE.get(use, 0)
    if want <= 0 or mem_available_mb is None:
        return 0
    room = int(mem_available_mb) - RAM_RESERVE_MB
    return max(0, min(want, room)) // STEP_MIB * STEP_MIB
