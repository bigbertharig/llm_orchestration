from __future__ import annotations

from rig_control.manager import campaigns, core
from test_manager import menv  # noqa: F401


def orchestration_preview(*, available: bool = True) -> dict:
    return {
        "schema": "rig-scheduling-preview/1",
        "revision": "sha256:test-revision",
        "observed_at": "2026-11-21T00:00:00+00:00",
        "resources": {"cards": [
            {"index": 0, "total_mb": 24576},
            {"index": 1, "total_mb": 6144},
            {"index": 2, "total_mb": 6144},
        ]},
        "workloads": [
            {"id": "model_a", "alternatives": [{
                "gpus": [1], "ctx": 8192, "placement": "gpu",
                "currently_available": available, "orchestration_preferred": True,
                "estimated_load_seconds": 10.0, "load_estimate_basis": "measured_layout",
            }]},
            {"id": "model_b", "alternatives": [{
                "gpus": [2], "ctx": 8192, "placement": "gpu",
                "currently_available": True, "orchestration_preferred": True,
                "estimated_load_seconds": 10.0, "load_estimate_basis": "fallback",
            }]},
        ],
    }


def campaign_request(**extra) -> dict:
    return {
        "name": "two_models",
        "blocks": [
            {"id": "a_reasoning", "model": "a", "suite": "bench-reasoning",
             "runtime_group": "model_a", "estimated_duration_s": 60},
            {"id": "a_knowledge", "model": "a", "suite": "bench-knowledge",
             "runtime_group": "model_a", "estimated_duration_s": 40},
            {"id": "b_reasoning", "model": "b", "suite": "bench-reasoning",
             "runtime_group": "model_b", "estimated_duration_s": 100},
        ],
        **extra,
    }


def test_simulator_serializes_loads_but_overlaps_disjoint_model_groups():
    campaign, workloads = campaigns.compile_campaign(campaign_request())
    assert [row["id"] for row in workloads] == ["model_a", "model_b"]
    result = campaigns.simulate(campaign, orchestration_preview())
    assert result["feasible"] is True
    assert result["hardware_authority"] == "orchestration"
    assert result["authoritative"] is False and result["admission_required_at_execution"] is True
    assert result["estimated_makespan_s"] == 120.0
    by_group = {row["group"]: row for row in result["timeline"]}
    assert by_group["model_a"]["load_start_s"] == 0.0
    assert by_group["model_a"]["run_start_s"] == 10.0
    assert by_group["model_b"]["load_start_s"] == 10.0
    assert by_group["model_b"]["run_start_s"] == 20.0
    assert [row["id"] for row in by_group["model_a"]["blocks"]] == ["a_reasoning", "a_knowledge"]


def test_live_simulation_refuses_currently_unavailable_alternatives():
    campaign, _ = campaigns.compile_campaign(campaign_request(planning_mode="live"))
    result = campaigns.simulate(campaign, orchestration_preview(available=False))
    assert result["feasible"] is False
    assert result["reason"] == "no live placement alternatives"
    assert result["unscheduled_groups"] == ["model_a"]


def test_host_ram_constraint_can_serialize_disjoint_gpu_groups():
    offered = orchestration_preview()
    offered["constraints"] = {"host_available_mb": 1000}
    for workload in offered["workloads"]:
        workload["alternatives"][0]["resources"] = {"host_ram_mb": 600}
    campaign, _ = campaigns.compile_campaign(campaign_request())
    result = campaigns.simulate(campaign, offered)
    by_group = {row["group"]: row for row in result["timeline"]}
    assert by_group["model_b"]["load_start_s"] == by_group["model_a"]["end_s"]
    assert result["estimated_makespan_s"] == 220.0


def test_dependencies_and_cycles_are_checked():
    request = campaign_request()
    request["blocks"][2]["depends_on"] = ["a_knowledge"]
    campaign, _ = campaigns.compile_campaign(request)
    result = campaigns.simulate(campaign, orchestration_preview())
    by_group = {row["group"]: row for row in result["timeline"]}
    assert by_group["model_b"]["load_start_s"] == by_group["model_a"]["end_s"]

    request["blocks"][0]["depends_on"] = ["b_reasoning"]
    campaign, _ = campaigns.compile_campaign(request)
    result = campaigns.simulate(campaign, orchestration_preview())
    assert result["feasible"] is False and result["reason"] == "dependency cycle"


def test_preview_api_asks_orchestration_and_returns_non_authoritative_plan(menv, monkeypatch):
    captured = {}

    async def fake_preview(self, workloads, candidate_limit=16):
        captured["workloads"] = workloads
        captured["candidate_limit"] = candidate_limit
        return orchestration_preview()

    monkeypatch.setattr(core.Manager, "schedule_preview", fake_preview)
    response = menv["client"].post("/v1/manager/benchmark-campaigns/preview",
                                   json={**campaign_request(), "candidate_limit": 8})
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["schema"] == "gateway-benchmark-campaign-preview/1"
    assert result["orchestration_revision"] == "sha256:test-revision"
    assert result["hardware_authority"] == "orchestration"
    assert [row["id"] for row in captured["workloads"]] == ["model_a", "model_b"]
    assert captured["candidate_limit"] == 8


def test_preview_validation_fails_before_calling_orchestration(menv, monkeypatch):
    async def should_not_run(*args, **kwargs):
        raise AssertionError("Orchestration must not be called")

    monkeypatch.setattr(core.Manager, "schedule_preview", should_not_run)
    response = menv["client"].post("/v1/manager/benchmark-campaigns/preview",
                                   json={"name": "bad", "blocks": [{"id": "x"}]})
    assert response.status_code == 400
    assert "needs a model" in response.json()["error"]["message"]
