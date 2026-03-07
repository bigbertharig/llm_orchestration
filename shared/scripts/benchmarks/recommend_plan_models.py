#!/usr/bin/env python3
"""Suggest llm_model fields for plan tasks using model_task_library.json."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


def parse_plan_tasks(plan_text: str) -> list[dict[str, str]]:
    sections = re.split(r"\n### ", plan_text)
    tasks: list[dict[str, str]] = []
    for sec in sections[1:]:
        lines = sec.strip().splitlines()
        if not lines:
            continue
        task_id = lines[0].strip()
        task = {"id": task_id, "task_class": "", "command": ""}
        for line in lines[1:]:
            s = line.strip()
            if s.startswith("- **task_class**:"):
                task["task_class"] = s.split(":", 1)[1].strip().lower()
            elif s.startswith("- **command**:"):
                m = re.search(r"`([^`]+)`", s)
                task["command"] = (m.group(1) if m else s.split(":", 1)[1].strip()).lower()
        tasks.append(task)
    return tasks


def choose_profile(task: dict[str, str], lib: dict[str, Any]) -> str:
    hay = f"{task.get('id', '').lower()} {task.get('command', '').lower()}"
    for rule in lib.get("keyword_profile_map", []):
        profile = str(rule.get("profile", "")).strip()
        for kw in rule.get("keywords", []):
            keyword = str(kw).strip().lower()
            if keyword and keyword in hay:
                return profile
    return "general_qa"


def profile_index(lib: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for entry in lib.get("task_profiles", []):
        pid = str(entry.get("id", "")).strip()
        if pid:
            out[pid] = entry
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Recommend model settings for llm tasks in a plan.")
    ap.add_argument("--plan", required=True, help="Path to plan.md")
    ap.add_argument("--library", default="model_task_library.json", help="Path to model task library JSON")
    ap.add_argument("--output", default="", help="Optional output JSON path")
    args = ap.parse_args()

    plan_path = Path(args.plan).expanduser().resolve()
    this_dir = Path(__file__).resolve().parent
    lib_path = (this_dir / args.library).resolve() if not Path(args.library).is_absolute() else Path(args.library)

    lib = json.loads(lib_path.read_text(encoding="utf-8"))
    profiles = profile_index(lib)
    tasks = parse_plan_tasks(plan_path.read_text(encoding="utf-8"))

    recs: list[dict[str, Any]] = []
    for task in tasks:
        if task.get("task_class") != "llm":
            continue
        profile = choose_profile(task, lib)
        entry = profiles.get(profile) or profiles.get("general_qa")
        if not entry:
            continue
        model = entry.get("recommended_model", {})
        recs.append(
            {
                "task_id": task.get("id"),
                "profile": profile,
                "recommendation": {
                    "llm_model": model.get("llm_model"),
                    "llm_min_tier": model.get("llm_min_tier"),
                    "llm_placement": model.get("llm_placement")
                },
                "evidence_test_ids": entry.get("evidence_test_ids", [])
            }
        )

    payload = {
        "plan": str(plan_path),
        "library": str(lib_path),
        "recommendations": recs
    }

    if args.output:
        out_path = Path(args.output).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Wrote recommendations: {out_path}")
    else:
        print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
