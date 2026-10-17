"""The remote profiles llm-gateway offers (`rig start remote <id>`).

llm-gateway owns the offer: its tools/generate_profiles.py resolves the offered models against the
model registry (benchmarking model_registry.json) and writes remote_profiles.json. On the rig that
file is the gateway's deployed copy; on the Pi it is the gateway repo. Each profile is already
resolved: id, model, gguf_path (archive), ctx_size, batch_size, parallel, extra_args, image,
gpus_needed, optional tensor_split.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

PATH = Path(os.getenv("RIG_REMOTE_PROFILES", "/home/bryan/llm-gateway/remote_profiles.json"))


def load(path: Path = PATH) -> dict[str, dict]:
    """Profiles by id. Raises OSError, ValueError or KeyError on a missing or malformed file."""
    items = json.loads(path.read_text(encoding="utf-8"))["remote_profiles"]
    return {str(p["id"]): p for p in items}
