"""Always-on hardware diagnostics must reflect the guard heartbeat and GPU scan."""

from orchestration_api import hardware


def test_sample_marks_fresh_guard_and_visible_gpus_healthy(monkeypatch):
    monkeypatch.setattr(hardware.time, "time", lambda: 100)
    replies = iter([{"level": "ok", "admit": True, "why": "ok",
                     "state": {"updated_ts": 99}},
                    {"mode": "idle", "gpus": {"ok": True, "ids": [0, 1], "errors": []}}])
    monkeypatch.setattr(hardware, "_rig_json", lambda *args: next(replies))
    row = hardware.sample()
    assert row["ok"] and row["gpu_ids"] == [0, 1]


def test_sample_rejects_stale_guard_or_missing_gpu(monkeypatch):
    monkeypatch.setattr(hardware.time, "time", lambda: 100)
    replies = iter([{"level": "ok", "state": {"updated_ts": 80}},
                    {"gpus": {"ok": False, "ids": [0], "errors": ["GPU 1 missing"]}}])
    monkeypatch.setattr(hardware, "_rig_json", lambda *args: next(replies))
    row = hardware.sample()
    assert not row["ok"] and row["gpu_errors"] == ["GPU 1 missing"]
