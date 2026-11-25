"""Pure benchmark campaign simulation over Orchestration placement alternatives."""

from __future__ import annotations

from typing import Any


CAMPAIGN_SCHEMA = "benchmark-campaign-plan/1"
PREVIEW_SCHEMA = "gateway-benchmark-campaign-preview/1"
ORCHESTRATION_SCHEMA = "rig-scheduling-preview/1"
DEFAULT_BLOCK_DURATION_S = 300.0
MAX_BLOCKS = 256
FIT_FIELDS = ("use", "profile", "ctx", "ctx_min", "context_mode", "margin_mb",
              "batch_size", "engine", "placement_policy")
BLOCK_FIELDS = {"id", "model", "suite", "depends_on", "runtime_group",
                "estimated_duration_s", *FIT_FIELDS}


def _host_slot(earliest: float, span: float, required_mb: int, capacity_mb: int | None,
               allocations: list[tuple[float, float, int]]) -> float | None:
    """Earliest allocation boundary where host RAM fits for the whole interval."""
    if required_mb <= 0:
        return earliest
    if capacity_mb is None or required_mb > capacity_mb:
        return None
    starts = sorted({earliest, *(end for _start, end, _mb in allocations if end >= earliest)})
    for start in starts:
        end = start + span
        events: dict[float, int] = {}
        for held_start, held_end, held_mb in allocations:
            if held_start >= end or held_end <= start:
                continue
            events[max(start, held_start)] = events.get(max(start, held_start), 0) + held_mb
            events[min(end, held_end)] = events.get(min(end, held_end), 0) - held_mb
        used = peak = 0
        for point in sorted(events):
            used += events[point]
            peak = max(peak, used)
        if peak + required_mb <= capacity_mb:
            return start
    return None


def compile_campaign(request: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Validate blocks, collapse runtime groups, and emit Orchestration workload intents."""
    if not isinstance(request, dict):
        raise ValueError("campaign preview request must be an object")
    allowed = {"schema", "name", "blocks", "planning_mode", "candidate_limit"}
    if set(request) - allowed:
        raise ValueError("unknown campaign preview field(s): " +
                         ", ".join(sorted(set(request) - allowed)))
    if request.get("schema", CAMPAIGN_SCHEMA) != CAMPAIGN_SCHEMA:
        raise ValueError(f"schema must be {CAMPAIGN_SCHEMA}")
    name = request.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("name must be nonempty text")
    mode = request.get("planning_mode", "after_current_work")
    if mode not in ("live", "after_current_work"):
        raise ValueError("planning_mode must be live or after_current_work")
    blocks = request.get("blocks")
    if not isinstance(blocks, list) or not blocks or len(blocks) > MAX_BLOCKS:
        raise ValueError(f"blocks must contain 1..{MAX_BLOCKS} items")

    normalized, ids = [], set()
    for index, raw in enumerate(blocks):
        if not isinstance(raw, dict) or set(raw) - BLOCK_FIELDS:
            unknown = sorted(set(raw) - BLOCK_FIELDS) if isinstance(raw, dict) else []
            raise ValueError(f"block {index} has unsupported fields: {unknown}")
        block_id, model = raw.get("id"), raw.get("model")
        if not isinstance(block_id, str) or not block_id or block_id in ids:
            raise ValueError(f"block {index} needs a unique nonempty id")
        if not isinstance(model, str) or not model:
            raise ValueError(f"block {block_id} needs a model")
        deps = raw.get("depends_on", [])
        if not isinstance(deps, list) or any(not isinstance(dep, str) or not dep for dep in deps):
            raise ValueError(f"block {block_id} depends_on must be a list of block ids")
        duration = raw.get("estimated_duration_s", DEFAULT_BLOCK_DURATION_S)
        if not isinstance(duration, (int, float)) or isinstance(duration, bool) or duration <= 0:
            raise ValueError(f"block {block_id} estimated_duration_s must be positive")
        ids.add(block_id)
        normalized.append({**raw, "depends_on": list(deps),
                           "estimated_duration_s": float(duration),
                           "duration_basis": "supplied" if "estimated_duration_s" in raw else "fallback"})
    for block in normalized:
        missing = sorted(set(block["depends_on"]) - ids)
        if missing:
            raise ValueError(f"block {block['id']} has unknown dependencies: {missing}")

    index_by_id = {block["id"]: index for index, block in enumerate(normalized)}
    group_by_block = {block["id"]: str(block.get("runtime_group") or block["id"])
                      for block in normalized}
    groups: dict[str, dict[str, Any]] = {}
    for block in normalized:
        group_id = group_by_block[block["id"]]
        group = groups.setdefault(group_id, {
            "id": group_id, "model": block["model"], "blocks": [], "depends_on": set(),
            "estimated_duration_s": 0.0,
            "fit": {key: block.get(key) for key in FIT_FIELDS if block.get(key) is not None},
        })
        if group["model"] != block["model"]:
            raise ValueError(f"runtime_group {group_id} contains different models")
        fit = {key: block.get(key) for key in FIT_FIELDS if block.get(key) is not None}
        if group["fit"] != fit:
            raise ValueError(f"runtime_group {group_id} changes model fit settings")
        group["blocks"].append(block)
        group["estimated_duration_s"] += block["estimated_duration_s"]
        for dep in block["depends_on"]:
            dep_group = group_by_block[dep]
            if dep_group == group_id:
                if index_by_id[dep] > index_by_id[block["id"]]:
                    raise ValueError(f"runtime_group {group_id} depends on a later member {dep}")
            else:
                group["depends_on"].add(dep_group)

    resolved_groups = []
    workloads = []
    for order, group in enumerate(groups.values()):
        group["depends_on"] = sorted(group["depends_on"])
        group["order"] = order
        resolved_groups.append(group)
        workloads.append({"id": group["id"], "model": group["model"],
                          "use": group["fit"].get("use", "benchmark"), **group["fit"]})
    candidate_limit = request.get("candidate_limit", 16)
    if type(candidate_limit) is not int or not 1 <= candidate_limit <= 64:
        raise ValueError("candidate_limit must be 1..64")
    campaign = {"schema": CAMPAIGN_SCHEMA, "name": name.strip(), "planning_mode": mode,
                "candidate_limit": candidate_limit, "groups": resolved_groups}
    return campaign, workloads


def simulate(campaign: dict[str, Any], orchestration: dict[str, Any]) -> dict[str, Any]:
    """Greedy deterministic discrete-event simulation; it never mutates rig state."""
    if orchestration.get("schema") != ORCHESTRATION_SCHEMA:
        raise ValueError(f"Orchestration preview schema must be {ORCHESTRATION_SCHEMA}")
    offered = {row.get("id"): row for row in orchestration.get("workloads", [])
               if isinstance(row, dict)}
    expected = {group["id"] for group in campaign["groups"]}
    if set(offered) != expected:
        raise ValueError("Orchestration workload alternatives do not match the campaign groups")

    card_mb = {int(card["index"]): int(card.get("total_mb") or 0)
               for card in (orchestration.get("resources") or {}).get("cards", [])}
    planning_mode = campaign["planning_mode"]
    unscheduled = {group["id"]: group for group in campaign["groups"]}
    completed_at: dict[str, float] = {}
    gpu_free_at = {gpu: 0.0 for gpu in card_mb}
    load_free_at = 0.0
    host_capacity = (orchestration.get("constraints") or {}).get("host_available_mb")
    host_capacity = int(host_capacity) if isinstance(host_capacity, (int, float)) else None
    host_allocations: list[tuple[float, float, int]] = []
    timeline = []

    while unscheduled:
        choices = []
        ready_groups = 0
        for group in unscheduled.values():
            if any(dep not in completed_at for dep in group["depends_on"]):
                continue
            ready_groups += 1
            dependency_end = max((completed_at[dep] for dep in group["depends_on"]), default=0.0)
            alternatives = offered[group["id"]].get("alternatives") or []
            if planning_mode == "live":
                alternatives = [row for row in alternatives if row.get("currently_available")]
            for candidate in alternatives:
                gpus = candidate.get("gpus") or []
                if not gpus or any(type(gpu) is not int or gpu not in gpu_free_at for gpu in gpus):
                    continue
                load_s = candidate.get("estimated_load_seconds")
                if not isinstance(load_s, (int, float)) or isinstance(load_s, bool) or load_s < 0:
                    continue
                earliest = max(dependency_end, load_free_at,
                               max((gpu_free_at[gpu] for gpu in gpus), default=0.0))
                host_mb = int((candidate.get("resources") or {}).get("host_ram_mb") or 0)
                span = float(load_s) + group["estimated_duration_s"]
                load_start = _host_slot(earliest, span, host_mb, host_capacity, host_allocations)
                if load_start is None:
                    continue
                run_start = load_start + float(load_s)
                end = run_start + group["estimated_duration_s"]
                scarce_mb = sum(card_mb[gpu] for gpu in gpus)
                score = (end, load_start, not bool(candidate.get("orchestration_preferred")),
                         len(gpus), scarce_mb, group["order"], tuple(gpus))
                choices.append((score, group, candidate, load_start, run_start, end, host_mb))
        if not choices:
            reason = ("dependency cycle" if ready_groups == 0 else
                      f"no {planning_mode} placement alternatives")
            return _result(campaign, orchestration, timeline, False, reason,
                           sorted(unscheduled), max(completed_at.values(), default=0.0))
        _, group, candidate, load_start, run_start, end, host_mb = min(choices, key=lambda row: row[0])
        gpus = candidate["gpus"]
        load_free_at = run_start
        for gpu in gpus:
            gpu_free_at[gpu] = end
        if host_mb:
            host_allocations.append((load_start, end, host_mb))
        completed_at[group["id"]] = end
        unscheduled.pop(group["id"])
        cursor = run_start
        blocks = []
        for block in group["blocks"]:
            block_end = cursor + block["estimated_duration_s"]
            blocks.append({"id": block["id"], "suite": block.get("suite"),
                           "start_s": round(cursor, 1), "end_s": round(block_end, 1),
                           "estimated_duration_s": block["estimated_duration_s"],
                           "duration_basis": block["duration_basis"]})
            cursor = block_end
        timeline.append({
            "group": group["id"], "model": group["model"], "gpus": gpus,
            "placement": candidate.get("placement"), "context": candidate.get("ctx"),
            "load_start_s": round(load_start, 1), "run_start_s": round(run_start, 1),
            "end_s": round(end, 1), "estimated_load_seconds": candidate["estimated_load_seconds"],
            "load_estimate_basis": candidate.get("load_estimate_basis"),
            "host_ram_mb": host_mb,
            "orchestration_preferred": bool(candidate.get("orchestration_preferred")),
            "blocks": blocks,
        })
    timeline.sort(key=lambda row: (row["load_start_s"], row["group"]))
    return _result(campaign, orchestration, timeline, True, None, [],
                   max(completed_at.values(), default=0.0))


def _result(campaign: dict[str, Any], orchestration: dict[str, Any], timeline: list[dict[str, Any]],
            feasible: bool, reason: str | None, unscheduled: list[str], makespan: float) -> dict[str, Any]:
    return {
        "schema": PREVIEW_SCHEMA,
        "campaign": campaign["name"],
        "planning_mode": campaign["planning_mode"],
        "feasible": feasible,
        **({"reason": reason} if reason else {}),
        "authoritative": False,
        "admission_required_at_execution": True,
        "hardware_authority": "orchestration",
        "orchestration_revision": orchestration.get("revision"),
        "orchestration_observed_at": orchestration.get("observed_at"),
        "estimated_makespan_s": round(makespan, 1),
        "timeline": timeline,
        "unscheduled_groups": unscheduled,
    }
