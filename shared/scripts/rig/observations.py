"""Rig-local load observations used to calibrate hardware fit.

The rig owns the history consumed by ``fit.py``. Gateway may send a load
receipt through the rig command interface after a completed or failed load.
Benchmarking can ingest these receipts as evidence when promoting a recipe.
Kinds:

- `load`: a ready load. Model, GGUF, micro-batch, use, the cards by UUID with their link and root port (the
  rig's scan), the manager's estimate per card, what llama.cpp allocated per card (the rig's `fit`), memory in
  use per card once serving (nvidia-smi), and the load time.
- `load_failed`: a load the rig refused or that never served, with the same facts, the error and whether it ran
  out of memory: the other side of the fit boundary.
- `placement` identifies GPU-only or RAM partial offload. RAM rows are retained as evidence but are excluded
  from full-GPU memory calibration and OOM ceilings.

- `depth_memory`: full-depth memory growth for a GGUF and card-size layout.
- `performance`: prefill and decode rates for a GGUF, runtime image, context and card set.

Rows without `kind` are loads. The file is rig-local runtime evidence, not a
second model-library authority.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

SHARED = Path(__file__).resolve().parents[2]
PATH: str | None = None  # override (tests); else RIG_FIT_OBSERVATIONS, else rig-local hardware state
KINDS = ("load", "load_failed", "depth_memory", "performance")
CARD_KEYS = ("index", "uuid", "name", "total_mb", "bus", "link_gen", "link_width", "root_port", "via")


def path() -> str:
    return PATH or os.getenv("RIG_FIT_OBSERVATIONS") or str(SHARED / "brain" / "hw" / "load_observations.jsonl")


def record(kind: str, **fields: Any) -> dict:
    if kind not in KINDS:
        raise ValueError(f"unknown observation kind {kind!r}; one of {KINDS}")
    row = {"kind": kind, "at": datetime.now(timezone.utc).isoformat(timespec="seconds"), **fields}
    p = path()
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, default=str) + "\n")
    return row


def rows(kind: str | None = None, model: str | None = None) -> Iterator[dict]:
    try:
        f = open(path(), encoding="utf-8")
    except OSError:
        return
    with f:
        for line in f:
            try:
                row = json.loads(line)
            except ValueError:
                continue
            row.setdefault("kind", "load")
            if (kind is None or row["kind"] == kind) and (model is None or row.get("model") == model):
                yield row


def history(kind: str = "load") -> Iterator[dict]:
    """Every recorded row of a kind, the same load counted once."""
    seen = set()
    for row in rows(kind):
        key = (row.get("source"), row.get("ubatch"), row.get("n_ctx") or row.get("planned_ctx"),
               tuple(row.get("gpus") or []), row.get("at") if kind != "load" else None)
        if key in seen:
            continue
        seen.add(key)
        yield row


def cards_from(obs: dict | None, gpus: list[int]) -> list[dict]:
    """The named cards as the rig's latest scan describes them (UUID, link, root port), in the given order.
    Empty when the rig's report has no scan (an older rig.py)."""
    scan = {int(c["index"]): c for c in ((obs or {}).get("gpu_scan") or {}).get("cards") or []}
    return [{k: scan[g].get(k) for k in CARD_KEYS} for g in gpus if g in scan]
