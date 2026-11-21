"""Read-only scheduling capabilities derived from rig-owned admission rules.

The preview is advice, not an allocation.  Every later load still goes through
the normal lease and model-fit admission path against current hardware state.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import statistics
from pathlib import Path
from typing import Any, Iterable

import model_fit


SCHEMA = "rig-scheduling-preview/1"
MAX_WORKLOADS = 64
DEFAULT_CANDIDATE_LIMIT = 16
MAX_CANDIDATE_LIMIT = 64
MAX_PLACEMENT_PROBES = 512
WORKLOAD_FIELDS = {
    "id", "model", "use", "profile", "ctx", "ctx_min", "context_mode",
    "margin_mb", "batch_size", "engine", "placement_policy",
}


def _revision(document: dict[str, Any]) -> str:
    encoded = json.dumps(document, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _load_seconds(rows: Iterable[dict[str, Any]], model: str, source: str,
                  gpu_ids: list[int]) -> tuple[float, str, int]:
    exact, model_rows = [], []
    for row in rows:
        if row.get("kind") != "load" or row.get("model") != model:
            continue
        seconds = row.get("seconds")
        if not isinstance(seconds, (int, float)) or seconds <= 0:
            continue
        model_rows.append(float(seconds))
        same_source = not source or row.get("source") in (None, source)
        if same_source and sorted(row.get("gpus") or []) == sorted(gpu_ids):
            exact.append(float(seconds))
    samples = exact or model_rows
    if samples:
        return round(float(statistics.median(samples)), 1), "measured_layout" if exact else "measured_model", len(samples)
    return 60.0, "fallback", 0


def _fit_options(workload: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in workload.items()
            if key in WORKLOAD_FIELDS and key != "id" and value is not None}


def _candidate(fit: dict[str, Any], *, currently_available: bool,
               rows: Iterable[dict[str, Any]], preferred: bool) -> dict[str, Any]:
    gpus = sorted(int(gpu) for gpu in fit.get("gpus") or [])
    load_s, basis, samples = _load_seconds(
        rows, str(fit.get("model") or ""), Path(str(fit.get("gguf_path") or "")).name, gpus)
    host_mb = fit.get("host_estimate_mb")
    if host_mb is None and fit.get("placement") == "strata":
        host_mb = (fit.get("strata") or {}).get("min_host_available_mb")
    if host_mb is None:
        host_mb = fit.get("host_mb")
    return {
        "gpus": gpus,
        "ctx": fit.get("ctx"),
        "placement": fit.get("placement", "gpu"),
        "backend": fit.get("backend", "llama"),
        "currently_available": currently_available,
        "orchestration_preferred": preferred,
        "resources": {
            "gpu_ids": gpus,
            **({"host_ram_mb": int(host_mb)} if isinstance(host_mb, (int, float)) else {}),
        },
        "estimated_load_seconds": load_s,
        "load_estimate_basis": basis,
        "load_estimate_samples": samples,
        "chosen_by": fit.get("chosen_by"),
    }


def build_preview(request: dict[str, Any], *, library: Any, cards: list[dict[str, Any]],
                  free: list[dict[str, Any]], unavailable: dict[int, str],
                  host_available: int | None, loads_admitted: bool, guard_level: str,
                  boot_id: str, plan_version: int,
                  mode: str, observed_at: str, load_history: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Return revisioned placement alternatives for a set of workload intents."""
    if not isinstance(request, dict):
        raise ValueError("schedule preview request must be an object")
    if set(request) - {"workloads", "candidate_limit"}:
        raise ValueError("unknown schedule preview field(s): " +
                         ", ".join(sorted(set(request) - {"workloads", "candidate_limit"})))
    workloads = request.get("workloads")
    if not isinstance(workloads, list) or not workloads or len(workloads) > MAX_WORKLOADS:
        raise ValueError(f"workloads must contain 1..{MAX_WORKLOADS} items")
    candidate_limit = request.get("candidate_limit", DEFAULT_CANDIDATE_LIMIT)
    if type(candidate_limit) is not int or not 1 <= candidate_limit <= MAX_CANDIDATE_LIMIT:
        raise ValueError(f"candidate_limit must be 1..{MAX_CANDIDATE_LIMIT}")
    ids = [item.get("id") if isinstance(item, dict) else None for item in workloads]
    if any(not isinstance(item, str) or not item for item in ids) or len(ids) != len(set(ids)):
        raise ValueError("every workload needs a unique nonempty id")

    library_snapshot = library.load_snapshot()
    state_basis = {
        "boot_id": boot_id,
        "plan_version": int(plan_version),
        "mode": mode,
        "model_library_sha256": library_snapshot.get("source_sha256"),
        "cards": [{key: card.get(key) for key in
                   ("index", "uuid", "name", "total_mb", "bus", "link_gen", "link_width")}
                  for card in cards],
        "free_gpu_ids": sorted(int(card["index"]) for card in free),
        "unavailable": {str(key): value for key, value in sorted(unavailable.items())},
        "host_available_mb": host_available,
        "loads_admitted": bool(loads_admitted),
        "guard_level": guard_level,
    }
    revision = _revision(state_basis)
    idle = [{**card, "used_mb": 0} for card in cards]
    free_ids = {int(card["index"]) for card in free}
    history = list(load_history)
    results = []

    for workload in workloads:
        if not isinstance(workload, dict) or set(workload) - WORKLOAD_FIELDS:
            unknown = sorted(set(workload) - WORKLOAD_FIELDS) if isinstance(workload, dict) else []
            raise ValueError(f"workload must be an object with supported fields; unknown: {unknown}")
        options = _fit_options(workload)
        desired_mode = "bench" if options.get("use", "benchmark") == "benchmark" else "remote"
        live_mode_compatible = mode in ("idle", desired_mode)
        preferred_fit = model_fit.preview(options, library=library, cards=cards, free=idle,
                                          unavailable={}, host_available=host_available)
        preferred_gpus = sorted(preferred_fit.get("gpus") or [])
        alternatives: dict[tuple, dict[str, Any]] = {}
        preferred_key = (tuple(preferred_gpus), preferred_fit.get("ctx"),
                         preferred_fit.get("placement"), preferred_fit.get("backend"))
        alternatives[preferred_key] = _candidate(
            {**preferred_fit, "model": options["model"]},
            currently_available=(bool(loads_admitted) and live_mode_compatible and
                                 set(preferred_gpus) <= free_ids),
            rows=history,
            preferred=True,
        )
        probes = 0
        search_truncated = False
        for count in range(1, len(cards) + 1):
            for combo in itertools.combinations(sorted(card["index"] for card in cards), count):
                if probes >= MAX_PLACEMENT_PROBES:
                    search_truncated = True
                    break
                probes += 1
                try:
                    fit = model_fit.preview({**options, "gpu_ids": list(combo)}, library=library,
                                            cards=cards, free=idle, unavailable={},
                                            host_available=host_available)
                except model_fit.ModelFitError as exc:
                    if exc.kind in ("invalid", "not_found"):
                        raise
                    continue
                gpu_ids = sorted(fit.get("gpus") or [])
                key = (tuple(gpu_ids), fit.get("ctx"), fit.get("placement"), fit.get("backend"))
                alternatives[key] = _candidate(
                    {**fit, "model": options["model"]},
                    currently_available=(bool(loads_admitted) and live_mode_compatible and
                                         set(gpu_ids) <= free_ids),
                    rows=history,
                    preferred=gpu_ids == preferred_gpus,
                )
            if search_truncated:
                break
        ordered = sorted(alternatives.values(), key=lambda row: (
            not row["orchestration_preferred"], len(row["gpus"]), row["gpus"],
            -(int(row.get("ctx") or 0)),
        ))
        results.append({
            "id": workload["id"],
            "model": options["model"],
            "alternatives": ordered[:candidate_limit],
            "alternatives_total": len(ordered),
            "alternatives_truncated": search_truncated or len(ordered) > candidate_limit,
            "placement_search_probes": probes,
        })

    return {
        "schema": SCHEMA,
        "revision": revision,
        "observed_at": observed_at,
        "authoritative": False,
        "admission_required_at_execution": True,
        "constraints": {
            "hardware_authority": "orchestration",
            "max_concurrent_model_loads": 1,
            "rig_mode": mode,
            "host_available_mb": host_available,
            "loads_admitted": bool(loads_admitted),
            "guard_level": guard_level,
        },
        "resources": {
            "cards": state_basis["cards"],
            "free_gpu_ids": state_basis["free_gpu_ids"],
            "unavailable": state_basis["unavailable"],
        },
        "model_library_sha256": state_basis["model_library_sha256"],
        "workloads": results,
    }
