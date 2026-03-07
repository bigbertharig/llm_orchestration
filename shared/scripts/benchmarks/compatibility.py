#!/usr/bin/env python3
"""Helpers for benchmark backend/test compatibility policy."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def default_status_path() -> Path:
    return Path(__file__).resolve().parent / "benchmark_status.json"


def load_status(path: Path | None = None) -> dict[str, Any]:
    status_path = (path or default_status_path()).expanduser().resolve()
    if not status_path.exists():
        return {"backend_notes": [], "certified_tests": [], "model_tokenizers": {}, "runtime_issues": []}
    data = json.loads(status_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        return {"backend_notes": [], "certified_tests": [], "model_tokenizers": {}, "runtime_issues": []}
    data.setdefault("backend_notes", [])
    data.setdefault("certified_tests", [])
    data.setdefault("model_tokenizers", {})
    data.setdefault("runtime_issues", [])
    return data


def save_status(data: dict[str, Any], path: Path | None = None) -> Path:
    status_path = (path or default_status_path()).expanduser().resolve()
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return status_path


def derive_backend_id(model_backend: str, model_args: str, apply_chat_template: bool = False) -> str:
    backend = str(model_backend or "").strip()
    args = str(model_args or "")
    if backend == "local-chat-completions" and "/v1/chat/completions" in args:
        return "ollama_chat_completions_templated" if apply_chat_template else "ollama_chat_completions_raw"
    if backend == "local-completions" and "/v1/completions" in args:
        # Port 11435 = proxy that flattens array prompts for MC/loglikelihood tasks
        if ":11435/" in args:
            return "ollama_completions_proxied"
        return "ollama_completions"
    return ""


def find_certified_test(
    status: dict[str, Any], backend_id: str, test_id: str
) -> dict[str, Any] | None:
    for row in status.get("certified_tests", []):
        if (
            str(row.get("backend_id", "")).strip() == backend_id
            and str(row.get("test_id", "")).strip() == test_id
        ):
            return row
    return None


def required_tokenizer(status: dict[str, Any], model_id: str) -> str:
    tokenizers = status.get("model_tokenizers", {})
    if not isinstance(tokenizers, dict):
        return ""
    value = tokenizers.get(model_id, "")
    return str(value).strip()


def set_certified_test(
    status: dict[str, Any],
    *,
    backend_id: str,
    test_id: str,
    task_name: str,
    state: str,
    note: str,
    model_id: str = "",
    observed_at: str = "",
) -> None:
    rows = status.setdefault("certified_tests", [])
    if not isinstance(rows, list):
        raise ValueError("certified_tests must be a list")
    updated = {
        "backend_id": backend_id,
        "test_id": test_id,
        "task_name": task_name,
        "state": state,
        "note": note,
        "model_id": model_id,
        "observed_at": observed_at,
    }
    for idx, row in enumerate(rows):
        if (
            str(row.get("backend_id", "")).strip() == backend_id
            and str(row.get("test_id", "")).strip() == test_id
        ):
            rows[idx] = updated
            return
    rows.append(updated)


def upsert_runtime_issue(
    status: dict[str, Any],
    *,
    issue_id: str,
    subject: str,
    state: str,
    observed_at: str,
    note: str,
) -> None:
    rows = status.setdefault("runtime_issues", [])
    if not isinstance(rows, list):
        raise ValueError("runtime_issues must be a list")
    updated = {
        "issue_id": issue_id,
        "subject": subject,
        "state": state,
        "observed_at": observed_at,
        "note": note,
    }
    for idx, row in enumerate(rows):
        if str(row.get("issue_id", "")).strip() == issue_id:
            rows[idx] = updated
            return
    rows.append(updated)
