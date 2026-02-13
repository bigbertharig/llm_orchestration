#!/usr/bin/env python3
"""Local web dashboard for orchestration status."""

import argparse
import json
import re
import subprocess
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


def load_config(config_path: Path) -> dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_shared_path(config_path: Path, config: dict[str, Any]) -> Path:
    shared = Path(config.get("shared_path", "../"))
    if shared.is_absolute():
        return shared
    return (config_path.resolve().parent / shared).resolve()


def load_json(path: Path) -> dict[str, Any] | None:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def iter_task_files(folder_path: Path):
    if not folder_path.exists():
        return
    for task_file in folder_path.glob("*.json"):
        if task_file.name.endswith(".heartbeat.json"):
            continue
        yield task_file


def task_sort_key(task: dict[str, Any]) -> str:
    return (
        task.get("last_attempt_at")
        or task.get("completed_at")
        or task.get("started_at")
        or task.get("created_at")
        or ""
    )


def to_task_view(task: dict[str, Any]) -> dict[str, Any]:
    result = task.get("result") if isinstance(task.get("result"), dict) else {}
    return {
        "task_id": task.get("task_id"),
        "name": task.get("name"),
        "task_class": task.get("task_class") or "-",
        "type": task.get("type"),
        "batch_id": task.get("batch_id"),
        "assigned_to": task.get("assigned_to"),
        "status": task.get("status"),
        "attempts": task.get("attempts") or task.get("retry_count") or 0,
        "created_at": task.get("created_at"),
        "started_at": task.get("started_at"),
        "completed_at": task.get("completed_at"),
        "error": (result.get("output") or task.get("blocked_reason") or "")[:220],
    }


def list_tasks(shared_path: Path) -> dict[str, list[dict[str, Any]]]:
    lanes = {
        "queue": [],
        "processing": [],
        "complete": [],
        "failed": [],
    }
    for lane in lanes:
        folder = shared_path / "tasks" / lane
        for task_file in iter_task_files(folder) or []:
            task = load_json(task_file)
            if task:
                lanes[lane].append(task)
        lanes[lane].sort(key=task_sort_key, reverse=True)
    return lanes


def list_private_tasks(shared_path: Path) -> list[dict[str, Any]]:
    private_dir = shared_path / "brain" / "private_tasks"
    rows: list[dict[str, Any]] = []
    for task_file in iter_task_files(private_dir) or []:
        task = load_json(task_file)
        if task:
            rows.append(task)
    rows.sort(key=task_sort_key, reverse=True)
    return rows


def heartbeat_age_seconds(last_updated: str | None) -> int | None:
    if not last_updated:
        return None
    try:
        dt = datetime.fromisoformat(last_updated)
        return int((datetime.now() - dt).total_seconds())
    except Exception:
        return None


def load_worker_rows(shared_path: Path, processing_tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_worker: dict[str, list[str]] = {}
    for t in processing_tasks:
        worker = t.get("assigned_to")
        if worker:
            by_worker.setdefault(worker, []).append(t.get("name") or t.get("task_id") or "(task)")

    rows: list[dict[str, Any]] = []

    for hb_file in sorted((shared_path / "gpus").glob("gpu_*/heartbeat.json")):
        hb = load_json(hb_file)
        if not hb:
            continue
        name = hb.get("name") or hb_file.parent.name
        held = by_worker.get(name, [])[:]
        for at in hb.get("active_tasks", []) or []:
            if isinstance(at, dict):
                n = at.get("task_name") or at.get("task_id")
                if n and n not in held:
                    held.append(n)
        rows.append({
            "name": name,
            "gpu_id": hb.get("gpu_id"),
            "type": "gpu",
            "state": hb.get("state", "?"),
            "host": hb.get("hostname", "-"),
            "updated_at": hb.get("last_updated"),
            "age_s": heartbeat_age_seconds(hb.get("last_updated")),
            "gpu_temp_c": hb.get("temperature_c"),
            "cpu_temp_c": hb.get("cpu_temp_c"),
            "gpu_util": hb.get("gpu_util_percent"),
            "power_w": hb.get("power_draw_w"),
            "vram_used_mb": hb.get("vram_used_mb"),
            "vram_total_mb": hb.get("vram_total_mb"),
            "holding": held,
        })

    for hb_file in sorted((shared_path / "cpus").glob("*/heartbeat.json")):
        hb = load_json(hb_file)
        if not hb:
            continue
        name = hb.get("name") or hb_file.parent.name
        held = by_worker.get(name, [])[:]
        active_task_id = hb.get("active_task_id")
        if active_task_id:
            exists = any(active_task_id in x for x in held)
            if not exists:
                held.append(active_task_id)
        rows.append({
            "name": name,
            "gpu_id": None,
            "type": "cpu",
            "state": hb.get("state", "?"),
            "host": hb.get("hostname", "-"),
            "updated_at": hb.get("last_updated"),
            "age_s": heartbeat_age_seconds(hb.get("last_updated")),
            "gpu_temp_c": None,
            "cpu_temp_c": hb.get("cpu_temp_c"),
            "gpu_util": None,
            "power_w": None,
            "vram_used_mb": None,
            "vram_total_mb": None,
            "holding": held,
        })

    rows.sort(key=lambda r: (r["type"], r["name"]))
    return rows


def load_brain_state(shared_path: Path) -> dict[str, Any]:
    return load_json(shared_path / "brain" / "state.json") or {}


def count_by_batch(tasks: dict[str, list[dict[str, Any]]], batch_id: str) -> dict[str, int]:
    return {
        "queue": sum(1 for t in tasks["queue"] if t.get("batch_id") == batch_id),
        "processing": sum(1 for t in tasks["processing"] if t.get("batch_id") == batch_id),
        "complete": sum(1 for t in tasks["complete"] if t.get("batch_id") == batch_id),
        "failed": sum(1 for t in tasks["failed"] if t.get("batch_id") == batch_id),
        "private": sum(1 for t in tasks["private"] if t.get("batch_id") == batch_id),
    }


def lane_view(items: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    return [to_task_view(t) for t in items[:limit]]


def extract_stage_item(name: str | None) -> tuple[str | None, str | None]:
    if not name:
        return None, None
    m = re.match(r"^(?P<stage>.+?)_(?P<item>(contact|chunk|item)_[0-9A-Za-z]+)$", name)
    if not m:
        return None, None
    return m.group("stage"), m.group("item")


def build_batch_chain(tasks_by_lane: dict[str, list[dict[str, Any]]], batch_id: str) -> dict[str, Any]:
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
    stages: set[str] = set()
    edges: set[tuple[str, str]] = set()

    for name, task in by_name.items():
        stage, item = extract_stage_item(name)
        if stage and item:
            stages.add(stage)
            item_stage_status.setdefault(item, {})[stage] = lane_of.get(name, "unknown")

            for dep in task.get("depends_on", []) or []:
                dep_stage, dep_item = extract_stage_item(dep)
                if dep_stage and (dep_item == item):
                    edges.add((dep_stage, stage))

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
        "rows": rows,
        "row_count": len(items),
    }


def summarize(shared_path: Path, config: dict[str, Any]) -> dict[str, Any]:
    lanes = list_tasks(shared_path)
    private_tasks = list_private_tasks(shared_path)
    lanes["private"] = private_tasks

    brain = load_brain_state(shared_path)
    active_batches = brain.get("active_batches", {}) if isinstance(brain.get("active_batches"), dict) else {}

    workers = load_worker_rows(shared_path, lanes["processing"])
    brain_gpu_ids = list(config.get("brain", {}).get("gpus", []))
    gpu_by_id = {}
    for w in workers:
        if w.get("type") == "gpu":
            gpu_by_id[w.get("gpu_id")] = w

    brain_gpus = []
    for gpu_id in brain_gpu_ids:
        matched = gpu_by_id.get(gpu_id)
        if matched:
            row = dict(matched)
            row["note"] = "heartbeat"
            brain_gpus.append(row)
        else:
            brain_gpus.append({
                "name": f"brain-gpu-{gpu_id}",
                "gpu_id": gpu_id,
                "type": "gpu",
                "state": "no_heartbeat",
                "host": "-",
                "updated_at": None,
                "age_s": None,
                "gpu_temp_c": None,
                "cpu_temp_c": None,
                "gpu_util": None,
                "power_w": None,
                "vram_used_mb": None,
                "vram_total_mb": None,
                "holding": [],
                "note": "configured brain GPU has no worker heartbeat",
            })

    batches: dict[str, Any] = {}
    batch_chains: dict[str, Any] = {}
    for batch_id, meta in active_batches.items():
        counts = count_by_batch(lanes, batch_id)
        batches[batch_id] = {
            "plan": meta.get("plan"),
            "started_at": meta.get("started_at"),
            "total_hint": meta.get("total_tasks"),
            "counts": counts,
        }
        batch_chains[batch_id] = build_batch_chain(lanes, batch_id)

    # Keep lane tables focused on currently active batches so historical
    # completes don't swamp the dashboard.
    active_batch_ids = set(active_batches.keys())
    if active_batch_ids:
        lane_source = {
            "queue": [t for t in lanes["queue"] if t.get("batch_id") in active_batch_ids],
            "processing": [t for t in lanes["processing"] if t.get("batch_id") in active_batch_ids],
            "private": [t for t in lanes["private"] if t.get("batch_id") in active_batch_ids],
            "complete": [t for t in lanes["complete"] if t.get("batch_id") in active_batch_ids],
            "failed": [t for t in lanes["failed"] if t.get("batch_id") in active_batch_ids],
        }
    else:
        lane_source = lanes

    return {
        "generated_at": datetime.now().isoformat(),
        "counts": {k: len(v) for k, v in lanes.items()},
        "active_batches": batches,
        "batch_chains": batch_chains,
        "workers": workers,
        "brain_gpus": brain_gpus,
        "lanes": {
            "queue": lane_view(lane_source["queue"], 50),
            "processing": lane_view(lane_source["processing"], 50),
            "private": lane_view(lane_source["private"], 80),
            "complete": lane_view(lane_source["complete"], 50),
            "failed": lane_view(lane_source["failed"], 50),
        },
    }


def run_shell(cmd: str, timeout_s: int = 120) -> dict[str, Any]:
    try:
        proc = subprocess.run(
            cmd,
            shell=True,
            executable="/bin/bash",
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        return {
            "ok": proc.returncode == 0,
            "returncode": proc.returncode,
            "stdout": proc.stdout[-4000:],
            "stderr": proc.stderr[-4000:],
            "cmd": cmd,
        }
    except subprocess.TimeoutExpired as e:
        return {
            "ok": False,
            "returncode": -1,
            "stdout": (e.stdout or "")[-2000:] if isinstance(e.stdout, str) else "",
            "stderr": (e.stderr or "")[-2000:] if isinstance(e.stderr, str) else "",
            "cmd": cmd,
            "error": f"timeout after {timeout_s}s",
        }


def discover_plans(shared_path: Path) -> list[str]:
    plans_dir = shared_path / "plans"
    out: list[str] = []
    if not plans_dir.exists():
        return out
    for p in sorted(plans_dir.iterdir()):
        if p.is_dir() and (p / "plan.md").exists():
            out.append(p.name)
    return out


def default_plan_config(shared_path: Path, plan_name: str) -> dict[str, Any]:
    if plan_name == "research_assistant":
        input_dir = shared_path / "plans" / "research_assistant" / "input"
        default_input = "/mnt/shared/plans/research_assistant/input/canucks.md"
        if input_dir.exists():
            md_files = sorted([p for p in input_dir.glob("*.md") if p.is_file()])
            if md_files:
                default_input = f"/mnt/shared/plans/research_assistant/input/{md_files[0].name}"
        return {
            "INPUT_FILE": default_input,
            "SEARCH_DEPTH": "basic",
            "OUTPUT_FORMAT": "both",
            "RUN_MODE": "fresh",
        }
    if plan_name == "video_zim_batch":
        return {
            "ZIM_PATH": "/mnt/shared/path/to/archive.zim",
            "SOURCE_ID": "source_name",
            "OUTPUT_FOLDER": "/mnt/shared/plans/video_zim_batch/output",
            "RUN_MODE": "fresh",
        }
    return {"RUN_MODE": "fresh"}


def plan_input_help(shared_path: Path, plan_name: str) -> list[dict[str, str]]:
    plan_md = shared_path / "plans" / plan_name / "plan.md"
    if not plan_md.exists():
        return []
    try:
        text = plan_md.read_text(encoding="utf-8")
    except Exception:
        return []

    lines = text.splitlines()
    in_inputs = False
    out: list[dict[str, str]] = []
    for line in lines:
        if line.strip().lower() == "## inputs":
            in_inputs = True
            continue
        if in_inputs and line.startswith("## "):
            break
        if not in_inputs:
            continue
        m = re.match(r"^\s*-\s+\*\*(.+?)\*\*:\s*(.+)\s*$", line)
        if not m:
            continue
        key = m.group(1).strip()
        desc = m.group(2).strip()
        options = ", ".join(re.findall(r"`([^`]+)`", desc))
        out.append({
            "key": key,
            "description": desc,
            "options": options,
        })
    return out


HTML = """<!doctype html>
<html>
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>LLM Orchestration Dashboard</title>
  <style>
    :root {
      --bg: #08131f;
      --panel: #0f2234;
      --panel2: #122a3f;
      --line: #24445f;
      --text: #eaf4ff;
      --muted: #9db4c8;
      --ok: #38d89c;
      --warn: #ffcc66;
      --bad: #ff6b6b;
      --cpu: #5db4ff;
      --gpu: #ffa94d;
      --llm: #f5a97f;
      --meta: #cba6f7;
      --script: #74c7ec;
      --taskcpu: #94e2d5;
    }
    body {
      margin: 0;
      font-family: \"IBM Plex Sans\", \"Segoe UI\", sans-serif;
      background: radial-gradient(1200px 600px at 10% 0%, #12304a 0%, var(--bg) 60%);
      color: var(--text);
    }
    .wrap { max-width: 1400px; margin: 0 auto; padding: 14px; }
    h1 { margin: 0 0 10px; font-size: 24px; }
    h3 { margin: 2px 0 10px; font-size: 16px; }
    .meta { color: var(--muted); margin-bottom: 10px; font-size: 13px; }
    .grid { display: grid; gap: 10px; }
    .g5 { grid-template-columns: repeat(5, minmax(0, 1fr)); }
    .g2 { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    .card {
      background: linear-gradient(180deg, var(--panel), var(--panel2));
      border: 1px solid var(--line);
      border-radius: 10px;
      padding: 10px 12px;
      overflow: hidden;
    }
    .k { color: var(--muted); font-size: 12px; }
    .v { font-size: 24px; font-weight: 700; line-height: 1.1; }
    .ok { color: var(--ok); }
    .warn { color: var(--warn); }
    .bad { color: var(--bad); }
    .section { margin-top: 10px; }
    table { width: 100%; border-collapse: collapse; font-size: 12px; table-layout: fixed; }
    th, td { text-align: left; padding: 5px 6px; border-bottom: 1px solid var(--line); vertical-align: top; word-break: break-word; }
    th { color: var(--muted); font-weight: 600; }
    tr:last-child td { border-bottom: 0; }
    .pill { display: inline-block; padding: 2px 7px; border-radius: 999px; font-size: 11px; border: 1px solid var(--line); }
    .pill.gpu { background: rgba(255,169,77,.15); color: var(--gpu); }
    .pill.cpu { background: rgba(93,180,255,.15); color: var(--cpu); }
    .pill.llm { background: rgba(245,169,127,.16); color: var(--llm); }
    .pill.meta { background: rgba(203,166,247,.16); color: var(--meta); }
    .pill.script { background: rgba(116,199,236,.16); color: var(--script); }
    .pill.taskcpu { background: rgba(148,226,213,.16); color: var(--taskcpu); }
    .mono { font-family: \"JetBrains Mono\", \"Consolas\", monospace; }
    .lane-title { display: flex; justify-content: space-between; align-items: baseline; }
    .lane-count { color: var(--muted); font-size: 12px; }
    .chain-box { margin-top: 10px; padding: 8px; border: 1px solid var(--line); border-radius: 8px; }
    .chain-head { display:flex; justify-content:space-between; align-items:center; gap:8px; margin-bottom:6px; }
    .chain-controls { display:flex; gap:6px; align-items:center; }
    .small-btn {
      background: #16324a;
      color: var(--text);
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 3px 8px;
      cursor: pointer;
      font-size: 11px;
    }
    .small-btn:disabled { opacity: 0.45; cursor: default; }
    .chip { display:inline-block; padding:2px 6px; border-radius:999px; font-size:11px; border:1px solid var(--line); }
    .chip.queue { color:#f9c74f; background:rgba(249,199,79,.15); }
    .chip.processing { color:#4cc9f0; background:rgba(76,201,240,.15); }
    .chip.private { color:#cba6f7; background:rgba(203,166,247,.15); }
    .chip.complete { color:#4ad66d; background:rgba(74,214,109,.15); }
    .chip.failed { color:#ff6b6b; background:rgba(255,107,107,.15); }
    .chip.missing { color:#9db4c8; background:rgba(157,180,200,.12); }
    .tabs { display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 10px; }
    .tab-btn {
      background: #16324a;
      color: var(--text);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 6px 10px;
      cursor: pointer;
      font-size: 12px;
    }
    .tab-btn.active {
      border-color: #4f9bd1;
      background: #204565;
    }
    @media (max-width: 1180px) {
      .g5, .g2 { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <div class=\"wrap\">
    <div class=\"tabs\" style=\"margin-bottom:8px;\">
      <a class=\"tab-btn active\" href=\"/\">Dashboard</a>
      <a class=\"tab-btn\" href=\"/controls\">Controls</a>
    </div>
    <h1>LLM Orchestration Dashboard</h1>
    <div class=\"meta\" id=\"meta\">Loading...</div>

    <div class=\"grid g5\" id=\"countCards\"></div>

    <div class=\"section card\">
      <h3>Active Batches</h3>
      <div id=\"batches\"></div>
      <div id=\"batchChains\"></div>
    </div>

    <div class=\"section card\">
      <h3>Brain GPU Status</h3>
      <div id=\"brainGpus\"></div>
    </div>

    <div class=\"section card\">
      <h3>Workers (status + held tasks)</h3>
      <div id=\"workers\"></div>
    </div>

    <div class=\"section card\">
      <div class=\"tabs\" id=\"laneTabs\"></div>
      <div id=\"laneTable\"></div>
    </div>
  </div>

  <script>
    const fmt = (x) => x === null || x === undefined || x === '' ? '-' : x;
    const laneOrder = ['queue', 'processing', 'private', 'complete', 'failed'];
    let activeLane = 'processing';
    const chainState = {};

    function typeBadge(type) {
      return `<span class=\"pill ${type}\">${type}</span>`;
    }

    function classBadge(cls) {
      const norm = (cls || '').toLowerCase();
      const map = { cpu: 'taskcpu', llm: 'llm', meta: 'meta', script: 'script' };
      const key = map[norm] || 'script';
      return `<span class=\"pill ${key}\">${fmt(cls)}</span>`;
    }

    function tempClass(cpu) {
      if (cpu === null || cpu === undefined) return '';
      if (cpu >= 85) return 'bad';
      if (cpu >= 75) return 'warn';
      return 'ok';
    }

    function table(headers, rows) {
      if (!rows.length) return '<div class=\"k\">(none)</div>';
      return `
        <table>
          <thead><tr>${headers.map(h => `<th>${h}</th>`).join('')}</tr></thead>
          <tbody>
            ${rows.map(r => `<tr>${r.map(c => `<td>${c}</td>`).join('')}</tr>`).join('')}
          </tbody>
        </table>
      `;
    }

    function renderCountCards(counts) {
      const items = [
        ['Queue', counts.queue, ''],
        ['Processing', counts.processing, 'ok'],
        ['Private', counts.private, 'warn'],
        ['Complete', counts.complete, 'ok'],
        ['Failed', counts.failed, counts.failed > 0 ? 'bad' : ''],
      ];
      return items.map(([k, v, cls]) => `
        <div class=\"card\"><div class=\"k\">${k}</div><div class=\"v ${cls}\">${v}</div></div>
      `).join('');
    }

    function renderTaskLane(targetId, items) {
      const rows = items.map(t => [
        classBadge(t.task_class),
        fmt(t.name),
        fmt(t.assigned_to),
        `<span class=\"mono\">${fmt(t.batch_id)}</span>`,
        fmt(t.attempts),
        `<span class=\"mono\">${fmt(t.error)}</span>`
      ]);
      document.getElementById(targetId).innerHTML = table(['Class', 'Task', 'Worker', 'Batch', 'Try', 'Error'], rows);
    }

    function renderLaneTabs(counts, lanes) {
      const labels = {
        queue: 'Queue',
        processing: 'Processing',
        private: 'Private',
        complete: 'Complete',
        failed: 'Failed'
      };
      const html = laneOrder.map(l => {
        const cls = l === activeLane ? 'tab-btn active' : 'tab-btn';
        const count = counts[l] ?? 0;
        return `<button class=\"${cls}\" data-lane=\"${l}\">${labels[l]} (${count})</button>`;
      }).join('');
      const tabs = document.getElementById('laneTabs');
      tabs.innerHTML = html;
      tabs.querySelectorAll('button[data-lane]').forEach(btn => {
        btn.addEventListener('click', () => {
          activeLane = btn.getAttribute('data-lane');
          renderLaneTabs(counts, lanes);
          renderTaskLane('laneTable', lanes[activeLane] || []);
        });
      });
      renderTaskLane('laneTable', lanes[activeLane] || []);
    }

    function laneChip(lane) {
      if (!lane || lane === '-') return '<span class=\"chip missing\">-</span>';
      return `<span class=\"chip ${lane}\">${lane}</span>`;
    }

    function renderBatchChains(data) {
      const perPage = 15;
      const out = [];
      const chains = data.batch_chains || {};
      Object.entries(chains).forEach(([batchId, chain]) => {
        const stages = chain.stage_order || [];
        if (!stages.length) return;
        const totalRows = chain.row_count || (chain.rows || []).length;
        if (!chainState[batchId]) {
          chainState[batchId] = { collapsed: false, page: 1 };
        }
        const st = chainState[batchId];
        const totalPages = Math.max(1, Math.ceil(totalRows / perPage));
        if (st.page > totalPages) st.page = totalPages;
        const start = (st.page - 1) * perPage;
        const end = start + perPage;

        const visibleRows = (chain.rows || []).slice(start, end).map(r => {
          const cols = [ `<span class=\"mono\">${r.item}</span>` ];
          stages.forEach(s => cols.push(laneChip((r.stages || {})[s])));
          return cols;
        });
        const collapsed = st.collapsed;
        const body = collapsed
          ? ''
          : `${table(['Item', ...stages], visibleRows)}
             <div class=\"k\">showing ${visibleRows.length} of ${totalRows}</div>`;

        out.push(`
          <div class=\"chain-box\" data-batch=\"${batchId}\">
            <div class=\"chain-head\">
              <div class=\"k\">${batchId} chain: ${stages.join(' -> ')}</div>
              <div class=\"chain-controls\">
                <button class=\"small-btn\" data-action=\"toggle\" data-batch=\"${batchId}\">${collapsed ? 'Expand' : 'Collapse'}</button>
                <button class=\"small-btn\" data-action=\"prev\" data-batch=\"${batchId}\" ${collapsed || st.page <= 1 ? 'disabled' : ''}>Prev</button>
                <span class=\"k\">Page ${st.page}/${totalPages}</span>
                <button class=\"small-btn\" data-action=\"next\" data-batch=\"${batchId}\" ${collapsed || st.page >= totalPages ? 'disabled' : ''}>Next</button>
              </div>
            </div>
            <div>${body}</div>
          </div>
        `);
      });
      const container = document.getElementById('batchChains');
      container.innerHTML = out.join('') || '<div class=\"k\">(no itemized dependency chains detected)</div>';
      container.querySelectorAll('button[data-action]').forEach(btn => {
        btn.addEventListener('click', () => {
          const action = btn.getAttribute('data-action');
          const batchId = btn.getAttribute('data-batch');
          const st = chainState[batchId] || { collapsed: false, page: 1 };
          const total = (chains[batchId]?.row_count) || 0;
          const pages = Math.max(1, Math.ceil(total / perPage));
          if (action === 'toggle') st.collapsed = !st.collapsed;
          if (action === 'prev' && st.page > 1) st.page -= 1;
          if (action === 'next' && st.page < pages) st.page += 1;
          chainState[batchId] = st;
          renderBatchChains(data);
        });
      });
    }

    async function refresh() {
      const res = await fetch('/api/status');
      const data = await res.json();

      document.getElementById('meta').textContent = `Updated ${new Date(data.generated_at).toLocaleTimeString()}`;
      document.getElementById('countCards').innerHTML = renderCountCards(data.counts);

      const batchRows = Object.entries(data.active_batches).map(([id, b]) => {
        const c = b.counts;
        const total = Math.max(b.total_hint || 0, c.queue + c.processing + c.complete + c.failed);
        return [
          `<span class=\"mono\">${id}</span>`,
          fmt(b.plan),
          `${c.complete}/${total}`,
          `${c.queue}/${c.processing}/${c.failed}/${c.private}`,
          fmt(b.started_at ? new Date(b.started_at).toLocaleTimeString() : '-')
        ];
      });
      document.getElementById('batches').innerHTML = table(
        ['Batch', 'Plan', 'Done', 'Q/P/F/Priv', 'Started'],
        batchRows
      );
      renderBatchChains(data);

      const brainRows = (data.brain_gpus || []).map(w => {
        const vram = (w.vram_used_mb !== null && w.vram_total_mb !== null)
          ? `${w.vram_used_mb}/${w.vram_total_mb}`
          : '-';
        return [
          fmt(w.gpu_id),
          fmt(w.name),
          fmt(w.state),
          fmt(w.gpu_temp_c),
          fmt(w.gpu_util),
          fmt(w.power_w),
          vram,
          `<span class=\"${tempClass(w.cpu_temp_c)}\">${fmt(w.cpu_temp_c)}</span>`,
          fmt(w.age_s),
          fmt(w.note)
        ];
      });
      document.getElementById('brainGpus').innerHTML = table(
        ['GPU', 'Name', 'State', 'GPU C', 'GPU %', 'W', 'VRAM MB', 'CPU C', 'HB s', 'Note'],
        brainRows
      );

      const workerRows = data.workers.map(w => [
        typeBadge(w.type),
        fmt(w.name),
        fmt(w.state),
        fmt(w.host),
        `<span class=\"${tempClass(w.cpu_temp_c)}\">${fmt(w.cpu_temp_c)}</span>`,
        fmt(w.gpu_temp_c),
        fmt(w.gpu_util),
        fmt(w.power_w),
        `<span class=\"mono\">${(w.holding || []).slice(0,2).join(' | ') || '-'}</span>`,
        fmt(w.age_s)
      ]);
      document.getElementById('workers').innerHTML = table(
        ['Type', 'Name', 'State', 'Host', 'CPU C', 'GPU C', 'GPU %', 'W', 'Holding', 'HB s'],
        workerRows
      );

      renderLaneTabs(data.counts, data.lanes);
    }

    refresh().catch(console.error);
    setInterval(() => refresh().catch(console.error), 2000);
  </script>
</body>
</html>
"""

CONTROLS_HTML = """<!doctype html>
<html>
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>LLM Orchestration Controls</title>
  <style>
    :root {
      --bg: #08131f;
      --panel: #0f2234;
      --panel2: #122a3f;
      --line: #24445f;
      --text: #eaf4ff;
      --muted: #9db4c8;
      --warn: #ffcc66;
      --bad: #ff6b6b;
      --ok: #38d89c;
    }
    body { margin: 0; font-family: \"IBM Plex Sans\", \"Segoe UI\", sans-serif; background: radial-gradient(1200px 600px at 10% 0%, #12304a 0%, var(--bg) 60%); color: var(--text); }
    .wrap { max-width: 1100px; margin: 0 auto; padding: 14px; }
    .tabs { display: flex; gap: 8px; margin-bottom: 10px; }
    .tab-btn { background: #16324a; color: var(--text); border: 1px solid var(--line); border-radius: 8px; padding: 6px 10px; text-decoration: none; }
    .tab-btn.active { border-color: #4f9bd1; background: #204565; }
    .card { background: linear-gradient(180deg, var(--panel), var(--panel2)); border: 1px solid var(--line); border-radius: 10px; padding: 12px; margin-bottom: 10px; }
    h1 { margin: 0 0 10px; font-size: 24px; }
    h3 { margin: 0 0 8px; font-size: 16px; }
    .row { display: flex; gap: 8px; flex-wrap: wrap; align-items: center; }
    .start-grid { display: grid; grid-template-columns: minmax(0, 1fr) minmax(320px, 420px); gap: 10px; align-items: start; }
    .field-grid { display: grid; gap: 8px; }
    .field-row { display: grid; grid-template-columns: 180px minmax(0, 1fr); gap: 8px; align-items: center; }
    .field-key { color: var(--muted); font-size: 12px; font-family: \"JetBrains Mono\", monospace; }
    .field-help { color: var(--muted); font-size: 12px; line-height: 1.35; }
    select, input, textarea, button {
      background: #0d1f31;
      color: var(--text);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 8px 10px;
      font-size: 13px;
    }
    button { cursor: pointer; background: #204565; }
    button.danger { background: #5a2323; border-color: #7a2d2d; }
    button.warn { background: #4d3b1a; border-color: #6a5120; }
    textarea { width: 100%; min-height: 120px; font-family: \"JetBrains Mono\", monospace; }
    #result { white-space: pre-wrap; font-family: \"JetBrains Mono\", monospace; font-size: 12px; color: var(--muted); }
    .k { color: var(--muted); font-size: 12px; }
    .ok { color: var(--ok); }
    .bad { color: var(--bad); }
    table { width: 100%; border-collapse: collapse; font-size: 12px; table-layout: fixed; }
    th, td { text-align: left; padding: 5px 6px; border-bottom: 1px solid var(--line); vertical-align: top; word-break: break-word; }
    th { color: var(--muted); font-weight: 600; }
    @media (max-width: 980px) {
      .start-grid { grid-template-columns: 1fr; }
      .field-row { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <div class=\"wrap\">
    <div class=\"tabs\">
      <a class=\"tab-btn\" href=\"/\">Dashboard</a>
      <a class=\"tab-btn active\" href=\"/controls\">Controls</a>
    </div>
    <h1>Control Panel</h1>

    <div class=\"card\">
      <h3>Kill Plan</h3>
      <div class=\"row\">
        <select id=\"killBatch\"></select>
        <button class=\"danger\" onclick=\"killPlan()\">Kill selected batch</button>
      </div>
      <div class=\"k\">Runs `kill_plan.py <batch_id>` and removes queued/processing/private tasks for that batch.</div>
    </div>

    <div class=\"card\">
      <h3>Return To Default</h3>
      <div class=\"row\">
        <button class=\"warn\" onclick=\"returnDefault()\">Reset to brain + 1 hot worker</button>
      </div>
      <div class=\"k\">Stops duplicate/orphan agents and restarts via startup defaults.</div>
    </div>

    <div class=\"card\">
      <h3>Start Plan</h3>
      <div class=\"row\" style=\"margin-bottom:8px;\">
        <select id=\"planName\"></select>
      </div>
      <div class=\"start-grid\">
        <div>
          <div class=\"k\">Config JSON</div>
          <textarea id=\"planConfig\">{
  "INPUT_FILE": "/mnt/shared/plans/research_assistant/input/canucks.md",
  "SEARCH_DEPTH": "basic",
  "OUTPUT_FORMAT": "both",
  "RUN_MODE": "fresh"
}</textarea>
          <div class=\"row\" style=\"margin-top:8px;\">
            <button onclick=\"startPlan()\">Submit plan</button>
          </div>
          <div class=\"k\">Uses `/mnt/shared/agents/submit.py` on GPU host.</div>
        </div>
        <div>
          <div class=\"k\">Input settings (with options)</div>
          <div id=\"planInputs\" style=\"margin-top:8px;\"></div>
        </div>
      </div>
    </div>

    <div class=\"card\">
      <h3>Result</h3>
      <div id=\"result\">(waiting)</div>
    </div>
  </div>

  <script>
    let planDefaults = {};
    let planInputs = {};

    async function api(path, payload) {
      const res = await fetch(path, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload || {})
      });
      return await res.json();
    }

    function showResult(data) {
      const ok = data.ok ? '[ok]' : '[error]';
      const text = [
        ok + ' ' + (data.message || ''),
        data.cmd ? ('cmd: ' + data.cmd) : '',
        data.stdout ? ('\\nstdout:\\n' + data.stdout) : '',
        data.stderr ? ('\\nstderr:\\n' + data.stderr) : '',
      ].filter(Boolean).join('\\n');
      const el = document.getElementById('result');
      el.textContent = text || '(no output)';
      el.className = data.ok ? 'ok' : 'bad';
    }

    async function refreshOptions() {
      const res = await fetch('/api/control/options');
      const data = await res.json();
      planDefaults = data.plan_defaults || {};
      planInputs = data.plan_inputs || {};
      const kill = document.getElementById('killBatch');
      kill.innerHTML = '';
      (data.active_batches || []).forEach(b => {
        const o = document.createElement('option');
        o.value = b;
        o.textContent = b;
        kill.appendChild(o);
      });
      if (!kill.options.length) {
        const o = document.createElement('option');
        o.value = '';
        o.textContent = '(no active batches)';
        kill.appendChild(o);
      }

      const plan = document.getElementById('planName');
      plan.innerHTML = '';
      (data.plans || []).forEach(p => {
        const o = document.createElement('option');
        o.value = p;
        o.textContent = p;
        plan.appendChild(o);
      });
      plan.onchange = () => applyPlanDefault(plan.value);
      if (plan.value) applyPlanDefault(plan.value);
    }

    function applyPlanDefault(planName) {
      const defaults = planDefaults[planName];
      if (defaults) {
        document.getElementById('planConfig').value = JSON.stringify(defaults, null, 2);
      }
      const help = {};
      (planInputs[planName] || []).forEach(x => { help[x.key] = x; });
      const src = defaults && Object.keys(defaults).length ? defaults : help;
      const keys = Object.keys(src);
      const rows = keys.map(key => {
        const h = help[key] || {};
        const opts = h.options ? `<div class=\"field-help\">Options: ${h.options}</div>` : '';
        const desc = h.description ? `<div class=\"field-help\">${h.description}</div>` : '';
        return `
          <div class=\"field-row\">
            <div class=\"field-key\">${key}</div>
            <div>${desc}${opts || '<div class=\"field-help\">(no explicit options listed)</div>'}</div>
          </div>
        `;
      }).join('');
      document.getElementById('planInputs').innerHTML = rows
        ? `<div class=\"field-grid\">${rows}</div>`
        : '<div class=\"k\">(no input metadata found)</div>';
    }

    async function killPlan() {
      const batchId = document.getElementById('killBatch').value;
      if (!batchId) return;
      showResult(await api('/api/control/kill_plan', { batch_id: batchId }));
      await refreshOptions();
    }

    async function returnDefault() {
      showResult(await api('/api/control/return_default', {}));
      await refreshOptions();
    }

    async function startPlan() {
      const planName = document.getElementById('planName').value;
      const configText = document.getElementById('planConfig').value;
      showResult(await api('/api/control/start_plan', { plan_name: planName, config_json: configText }));
      await refreshOptions();
    }

    refreshOptions().catch(console.error);
  </script>
</body>
</html>
"""


class DashboardHandler(BaseHTTPRequestHandler):
    shared_path: Path
    config: dict[str, Any]

    def _send_json(self, data: dict[str, Any], status: int = HTTPStatus.OK) -> None:
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self, html: str) -> None:
        body = html.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json_body(self) -> dict[str, Any]:
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            length = 0
        raw = self.rfile.read(length) if length > 0 else b"{}"
        try:
            data = json.loads(raw.decode("utf-8") or "{}")
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    def _gpu_ssh(self, remote_cmd: str, timeout_s: int = 180) -> dict[str, Any]:
        cmd = f"ssh -o BatchMode=yes gpu '{remote_cmd}'"
        return run_shell(cmd, timeout_s=timeout_s)

    def _control_options(self) -> dict[str, Any]:
        brain = load_brain_state(self.shared_path)
        active = brain.get("active_batches", {}) if isinstance(brain.get("active_batches"), dict) else {}
        plans = discover_plans(self.shared_path)
        plan_defaults = {p: default_plan_config(self.shared_path, p) for p in plans}
        plan_inputs = {p: plan_input_help(self.shared_path, p) for p in plans}
        return {
            "ok": True,
            "active_batches": sorted(active.keys()),
            "plans": plans,
            "plan_defaults": plan_defaults,
            "plan_inputs": plan_inputs,
        }

    def _kill_plan(self, batch_id: str) -> dict[str, Any]:
        if not batch_id:
            return {"ok": False, "message": "batch_id is required"}
        local_kill = Path(__file__).resolve().parent / "kill_plan.py"
        cmd = f"python3 {local_kill} {batch_id} --keep-workers --keep-models --no-default-warm"
        out = run_shell(cmd, timeout_s=240)
        out["message"] = f"Killed batch {batch_id}" if out.get("ok") else f"Failed to kill batch {batch_id}"
        return out

    def _return_default(self) -> dict[str, Any]:
        # Keep this explicit and deterministic: one startup owner from GPU host.
        remote_cmd = (
            "pkill -f '/mnt/shared/agents/brain.py' || true; "
            "pkill -f '/mnt/shared/agents/gpu.py' || true; "
            "pkill -f '/mnt/shared/agents/startup.py' || true; "
            "sleep 2; "
            "nohup /home/bryan/ml-env/bin/python /mnt/shared/agents/startup.py --config /mnt/shared/agents/config.json "
            ">> /mnt/shared/logs/startup-manual.log 2>&1 < /dev/null & "
            "sleep 4; "
            "pgrep -af '/mnt/shared/agents/startup.py|/mnt/shared/agents/brain.py|/mnt/shared/agents/gpu.py' || true"
        )
        out = self._gpu_ssh(remote_cmd, timeout_s=240)
        out["message"] = "Returned system to default startup state"
        return out

    def _start_plan(self, plan_name: str, config_json: str) -> dict[str, Any]:
        if not plan_name:
            return {"ok": False, "message": "plan_name is required"}
        plans = set(discover_plans(self.shared_path))
        if plan_name not in plans:
            return {"ok": False, "message": f"unknown plan: {plan_name}"}

        try:
            cfg_obj = json.loads(config_json) if config_json.strip() else {}
            if not isinstance(cfg_obj, dict):
                raise ValueError("config JSON must be an object")
        except Exception as e:
            return {"ok": False, "message": f"invalid config JSON: {e}"}

        cfg_inline = json.dumps(cfg_obj).replace("'", "'\"'\"'")
        remote_cmd = (
            f"python3 /mnt/shared/agents/submit.py /mnt/shared/plans/{plan_name} "
            f"--config '{cfg_inline}'"
        )
        out = self._gpu_ssh(remote_cmd, timeout_s=120)
        out["message"] = f"Submitted plan {plan_name}" if out.get("ok") else f"Failed to submit plan {plan_name}"
        return out

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/status":
            self._send_json(summarize(self.shared_path, self.config))
            return
        if parsed.path == "/api/control/options":
            self._send_json(self._control_options())
            return
        if parsed.path in ("/", "/index.html"):
            self._send_html(HTML)
            return
        if parsed.path == "/controls":
            self._send_html(CONTROLS_HTML)
            return
        self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        payload = self._read_json_body()

        if parsed.path == "/api/control/kill_plan":
            self._send_json(self._kill_plan(str(payload.get("batch_id", "")).strip()))
            return
        if parsed.path == "/api/control/return_default":
            self._send_json(self._return_default())
            return
        if parsed.path == "/api/control/start_plan":
            self._send_json(
                self._start_plan(
                    str(payload.get("plan_name", "")).strip(),
                    str(payload.get("config_json", "")).strip(),
                )
            )
            return

        self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def log_message(self, fmt: str, *args: Any) -> None:
        return


def main() -> None:
    parser = argparse.ArgumentParser(description="Run local orchestration dashboard")
    default_config = Path(__file__).resolve().parent.parent / "shared" / "agents" / "config.json"
    parser.add_argument("--config", default=str(default_config), help="Path to config.json")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8787)
    args = parser.parse_args()

    config_path = Path(args.config)
    config = load_config(config_path)
    shared_path = resolve_shared_path(config_path, config)

    class BoundHandler(DashboardHandler):
        pass

    BoundHandler.shared_path = shared_path
    BoundHandler.config = config

    server = ThreadingHTTPServer((args.host, args.port), BoundHandler)
    print(f"Dashboard: http://{args.host}:{args.port}")
    print(f"Shared path: {shared_path}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
