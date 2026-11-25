"""Read-only public inspection contracts; no leases, launches or repairs."""
from __future__ import annotations

from datetime import datetime, timezone
import json
import math
from pathlib import Path
import subprocess

import hw_power
from model_paths import resolve_hotset_gguf

SCHEMAS = {"power": "rig-power-check/1", "model-path": "rig-model-path/1"}


class InspectionError(ValueError):
    def __init__(self, kind: str, message: str):
        super().__init__(message)
        self.kind = kind


def gpu_ids(value: str) -> list[int]:
    pieces = value.split(",")
    if not pieces or any(not part.strip().isdigit() for part in pieces):
        raise InspectionError("invalid", "GPU selection needs distinct nonnegative integer IDs")
    ids = [int(part) for part in pieces]
    if len(ids) != len(set(ids)):
        raise InspectionError("invalid", "GPU selection contains duplicate IDs")
    return sorted(ids)


def check_power(selected: list[int]) -> dict:
    rules = hw_power.load_rules()
    rows = hw_power.cards()
    cards = {row["index"]: row for row in rows}
    if len(cards) != len(rows):
        raise InspectionError("unavailable", "GPU telemetry contains duplicate IDs")
    missing = sorted(set(selected) - cards.keys())
    if missing:
        raise InspectionError("not_found", f"GPU IDs absent from telemetry: {missing}")
    checks = []
    for gpu in selected:
        card = cards[gpu]
        values = [card[key] for key in ("limit_w", "default_w", "min_w", "max_w")]
        if any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0 for v in values):
            raise InspectionError("unavailable", f"GPU {gpu} has invalid power telemetry")
        planned = hw_power.planned(card, rules)
        if not math.isfinite(planned) or planned <= 0:
            raise InspectionError("invalid", f"GPU {gpu} has an invalid power plan")
        checks.append({"gpu_id": gpu, "actual_w": card["limit_w"], "planned_w": planned,
                       "matches": abs(card["limit_w"] - planned) < 0.5})
    ok = all(row["matches"] for row in checks)
    return {"ok": ok, "gpu_ids": selected, "checks": checks,
            **({} if ok else {"kind": "power_mismatch", "error": "GPU power limits differ from the configured plan"})}


def model_path(requested: str, config_path: Path) -> dict:
    if not requested or not Path(requested).is_absolute():
        raise InspectionError("invalid", "Model path must be absolute")
    config = json.loads(config_path.read_text()) if config_path.is_file() else {}
    if not isinstance(config, dict):
        raise InspectionError("invalid", "Site configuration must be an object")
    resolved = resolve_hotset_gguf(requested, config)
    if not resolved.is_file():
        raise InspectionError("not_found", "Model file is unavailable on the runtime host")
    return {"ok": True, "requested_path": requested, "resolved_path": str(resolved.resolve()),
            "source": "site_path_resolution"}


def inspect(action: str, *, gpus: str = "", path: str = "", config_path: Path) -> tuple[dict, int]:
    envelope = {"schema": SCHEMAS[action],
                "observed_at": datetime.now(timezone.utc).isoformat()}
    try:
        result = check_power(gpu_ids(gpus)) if action == "power" else model_path(path, config_path)
    except InspectionError as exc:
        result = {"ok": False, "kind": exc.kind, "error": str(exc)}
    except (OSError, subprocess.SubprocessError) as exc:
        result = {"ok": False, "kind": "unavailable", "error": f"Inspection unavailable ({type(exc).__name__})"}
    except (ValueError, TypeError, KeyError) as exc:
        result = {"ok": False, "kind": "invalid", "error": f"Invalid inspection data ({type(exc).__name__})"}
    return {**envelope, **result}, 0 if result["ok"] else 1
