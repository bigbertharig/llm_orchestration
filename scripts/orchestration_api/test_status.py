"""Current status and bounded job activity contract."""

import json

import pytest

from orchestration_api.status import activity, diagnostics, snapshot


def test_current_status_and_job_activity(tmp_path, monkeypatch):
    monkeypatch.setattr("orchestration_api.hardware.current", lambda: {"ok": True})
    shared = tmp_path
    (shared / "brain").mkdir()
    (shared / "heartbeats").mkdir()
    (shared / "gpus" / "gpu_0").mkdir(parents=True)
    batch_dir = shared / "plans" / "demo" / "history" / "batch-1"
    (batch_dir / "logs").mkdir(parents=True)
    (shared / "brain" / "state.json").write_text(json.dumps({"active_batches": {
        "batch-1": {"plan": "demo", "batch_dir": str(batch_dir)}}}))
    (shared / "heartbeats" / "brain.json").write_text(json.dumps({"name": "brain", "loaded_model": "m:1"}))
    (shared / "gpus" / "gpu_0" / "heartbeat.json").write_text(json.dumps({"name": "gpu-0", "gpu_id": 0,
                                                                       "loaded_model": "m:2"}))
    journal = batch_dir / "logs" / "batch_events.jsonl"
    journal.write_text("".join(json.dumps({"batch_id": "batch-1", "event": f"task_{n}",
                                           "task_name": f"work-{n}", "error": "private"}) + "\n"
                               for n in range(3)))
    status = snapshot(shared, {"gpus": [{"id": 0, "name": "gpu-0"}]})
    assert status["active_batch_count"] == 1
    assert status["workers"][0]["loaded_model"] == "m:2"
    assert diagnostics(shared, {"gpus": [{"id": 0, "name": "gpu-0"}]})["ok"]
    first = activity(shared, "batch-1", limit=2)
    assert [row["event"] for row in first["events"]] == ["task_0", "task_1"]
    assert first["has_more"] and "private" not in json.dumps(first)
    second = activity(shared, "batch-1", after=first["next_after"])
    assert [row["event"] for row in second["events"]] == ["task_2"]
    assert not second["has_more"]
    with pytest.raises(ValueError):
        activity(shared, "unknown")


def test_idle_task_heartbeats_do_not_mark_hardware_unhealthy(tmp_path, monkeypatch):
    monkeypatch.setattr("orchestration_api.hardware.current", lambda: {"ok": True, "gpu_ids": [0]})
    status = snapshot(tmp_path, {"gpus": [{"id": 0, "name": "gpu-0"}]})
    assert status["task_mode"] == "idle" and status["brain"] is None and status["workers"] == []
    assert status["visible_gpu_count"] == 1
    assert diagnostics(tmp_path, {"gpus": [{"id": 0, "name": "gpu-0"}]})["ok"]
    monkeypatch.setattr("orchestration_api.hardware.current", lambda: {"ok": False, "error": "rig unreachable"})
    assert not diagnostics(tmp_path, {})["ok"]
