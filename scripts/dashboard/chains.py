"""Batch chain building functions."""

import re
from typing import Any


def extract_stage_item(name: str | None) -> tuple[str | None, str | None]:
    """Extract stage and item from task name like 'stage_contact_0019'."""
    if not name:
        return None, None
    m = re.match(
        r"^(?P<stage>.+?)_(?P<item>(contact|chunk|item|prospect|person|day)_[0-9A-Za-z]+)$",
        name,
    )
    if not m:
        return None, None
    return m.group("stage"), m.group("item")


def build_batch_chain(tasks_by_lane: dict[str, list[dict[str, Any]]], batch_id: str) -> dict[str, Any]:
    """Build batch chain visualization data structure."""
    # Flatten all task lanes for this batch into one lookup by name.
    by_name: dict[str, dict[str, Any]] = {}
    lane_of: dict[str, str] = {}
    for lane, tasks in tasks_by_lane.items():
        for t in tasks:
            if t.get("batch_id") != batch_id:
                continue
            name = t.get("name")
            if not name:
                continue
            by_name[name] = t
            lane_of[name] = lane

    # Build item-stage status map from task names like stage_contact_0019.
    item_stage_status: dict[str, dict[str, str]] = {}
    stage_types: dict[str, str] = {}
    stages: set[str] = set()
    edges: set[tuple[str, str]] = set()
    global_edges: set[tuple[str, str]] = set()

    for name, task in by_name.items():
        stage, item = extract_stage_item(name)
        if stage and item:
            stages.add(stage)
            item_stage_status.setdefault(item, {})[stage] = lane_of.get(name, "unknown")
            executor = str(task.get("executor", "worker")).lower()
            task_class = str(task.get("task_class", "")).lower()
            if executor == "brain":
                stage_type = "brain"
            elif task_class == "script":
                stage_type = "gpu"
            elif task_class in {"cpu", "llm", "meta"}:
                stage_type = task_class
            else:
                stage_type = "-"
            if stage_type != "-":
                prev = stage_types.get(stage)
                # Keep strongest signal if mixed data appears.
                if prev is None or prev == "-" or stage_type == "brain":
                    stage_types[stage] = stage_type

            for dep in task.get("depends_on", []) or []:
                dep_stage, dep_item = extract_stage_item(dep)
                if dep_stage and (dep_item == item):
                    edges.add((dep_stage, stage))
        else:
            for dep in task.get("depends_on", []) or []:
                if dep in by_name:
                    global_edges.add((dep, name))

    # Topological-ish order from per-item edges. Fallback to alpha.
    if stages:
        indeg = {s: 0 for s in stages}
        children = {s: set() for s in stages}
        for a, b in edges:
            if a in stages and b in stages and b not in children[a]:
                children[a].add(b)
                indeg[b] += 1
        ready = sorted([s for s in stages if indeg[s] == 0])
        ordered = []
        while ready:
            s = ready.pop(0)
            ordered.append(s)
            for ch in sorted(children[s]):
                indeg[ch] -= 1
                if indeg[ch] == 0:
                    ready.append(ch)
            ready.sort()
        if len(ordered) != len(stages):
            stage_order = sorted(stages)
        else:
            stage_order = ordered
    else:
        # Fallback chain view for global stages (no per-item suffix yet),
        # so users still see declared order like build_strategy -> execute_searches.
        global_nodes = {
            n
            for n in by_name.keys()
            if n
            and n != "batch_summary"
            and not n.startswith("load_llm")
            and not n.startswith("load_worker_model")
        }
        if global_nodes:
            indeg = {s: 0 for s in global_nodes}
            children = {s: set() for s in global_nodes}
            for a, b in global_edges:
                if a in global_nodes and b in global_nodes and b not in children[a]:
                    children[a].add(b)
                    indeg[b] += 1
            ready = sorted([s for s in global_nodes if indeg[s] == 0])
            ordered = []
            while ready:
                s = ready.pop(0)
                ordered.append(s)
                for ch in sorted(children[s]):
                    indeg[ch] -= 1
                    if indeg[ch] == 0:
                        ready.append(ch)
                ready.sort()
            stage_order = ordered if len(ordered) == len(global_nodes) else sorted(global_nodes)
            for s in stage_order:
                t = by_name.get(s, {})
                executor = str(t.get("executor", "worker")).lower()
                task_class = str(t.get("task_class", "")).lower()
                if executor == "brain":
                    stage_types[s] = "brain"
                elif task_class == "script":
                    stage_types[s] = "gpu"
                elif task_class in {"cpu", "llm", "meta"}:
                    stage_types[s] = task_class
                else:
                    stage_types[s] = "-"
        else:
            stage_order = []

    items = sorted(item_stage_status.keys())
    rows = []
    for item in items:
        rows.append({
            "item": item,
            "stages": {s: item_stage_status[item].get(s, "-") for s in stage_order},
        })

    return {
        "stage_order": stage_order,
        "stage_types": {s: stage_types.get(s, "-") for s in stage_order},
        "rows": rows,
        "row_count": len(items),
    }
