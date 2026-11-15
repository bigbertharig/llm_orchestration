"""The executor consumes a selected plan without preparing it."""

from orchestration_api.controls import Controls


def test_prepared_submission_invokes_gpu_submit(tmp_path, monkeypatch):
    plan = tmp_path / "plans" / "arms" / "sample"
    plan.mkdir(parents=True)
    (plan / "plan.md").write_text("# Sample\n", encoding="utf-8")
    calls = []

    class Result:
        returncode = 0
        stdout = "batch submitted"
        stderr = ""

    monkeypatch.setattr("orchestration_api.controls.subprocess.run",
                        lambda cmd, **kwargs: calls.append((cmd, kwargs)) or Result())
    result = Controls(tmp_path, {}).submit_prepared("sample", "arms", "plan.md", {"RUN_MODE": "fresh"})

    assert result["ok"]
    assert len(calls) == 1
    assert "submit.py /mnt/shared/plans/arms/sample --plan-file plan.md" in calls[0][1]["input"]
    assert "RUN_MODE" not in calls[0][1]["input"]
    assert Controls(tmp_path, {}).submit_prepared("../other", "arms", "", {})["ok"] is False
