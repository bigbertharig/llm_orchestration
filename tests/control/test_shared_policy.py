import asyncio

import json

from pathlib import Path

import httpx

from fastapi import FastAPI

from fastapi.testclient import TestClient

from fastapi.responses import JSONResponse



from rig_control.manager.routes import router as manager_router

from rig_control.manager.errors import ManagerError



class SharedRig:
    def __init__(self, ready=True):
        self.ready = ready
        self.chats = [{"chat_id": "main", "gpu": 0, "context_cap": 262144, "used_tokens": 0}]
        self.calls = []

    async def status(self):
        return {"profiles": [], "shared_kv_service": ({"status": "ready",
                "model": "qwen3.6:35b-a3b-xl", "port": 11524, "mode": "gpu0-main-q4",
                "instance_id": "qwen-load-1",
                "chats": [dict(row) for row in self.chats]} if self.ready else None)}

    async def shared_kv(self, action, *, model=None, chat_id=None, message=None, max_tokens=None,
                        context_tokens=None):
        self.calls.append((action, chat_id, message, max_tokens))
        if action == "start":
            self.ready = True
            return {"status": "ready"}
        if action == "attach":
            capacity = context_tokens or 262144
            reservation = {"reservation_id": "reservation-" + chat_id,
                           "service_instance_id": "qwen-load-1",
                           "context_capacity": capacity, "occupied_tokens": 0,
                           "placement": {"gpus": [1]}, "reserved_mib": 1537,
                           "reservation_scope": "gpu_exclusive", "claim": "worker"}
            row = {"chat_id": chat_id, "gpu": 1,
                   "context_cap": capacity, "used_tokens": 0,
                   "reservation": reservation}
            self.chats.append(row)
            return {**row, "service_instance_id": "qwen-load-1"}
        if action == "detach":
            row = next(row for row in self.chats if row["chat_id"] == chat_id)
            self.chats = [row for row in self.chats if row["chat_id"] != chat_id]
            return {"detached": chat_id, "released_reservation": row.get("reservation")}
        if action == "turn":
            row = next(row for row in self.chats if row["chat_id"] == chat_id)
            row["used_tokens"] += 32
            return {"content": json.dumps({"name": "delegate_task", "arguments": {
                    "agent_type": "coding", "task": "Compute 7 times 6"}}),
                    "generated_tokens": 20, "used_tokens": row["used_tokens"],
                    "finish_reason": "stop"}
        raise AssertionError(action)

def test_manager_spawns_child_of_loaded_nemo_and_releases_it(monkeypatch):
    model = "nemotron-3.5-bolt:74b-a9b"
    class NemoRig(SharedRig):
        async def status(self):
            return {"shared_kv_service": {"status": "ready", "model": model,
                    "instance_id": "load-1", "chats": [dict(row) for row in self.chats],
                    "worker_capacity": {"status": "observed", "context_capacity": 262144,
                                        "available_chat_slots": 5 - len(self.chats),
                                        "free_gpu_count": 1 if len(self.chats) == 1 else 0,
                                        "free_gpu_ids": [1] if len(self.chats) == 1 else []}}}

        async def shared_kv(self, action, *, model=None, chat_id=None, message=None, max_tokens=None,
                            context_tokens=None):
            assert model == "nemotron-3.5-bolt:74b-a9b"
            result = await super().shared_kv(action, chat_id=chat_id, message=message,
                                             max_tokens=max_tokens, context_tokens=context_tokens)
            if action == "attach":
                result["service_instance_id"] = "load-1"
                result["reservation"]["service_instance_id"] = "load-1"
            return result

    monkeypatch.setattr("rig_control.manager.routes.models.catalog", lambda: {
        "models": {model: {"status": "active"}},
        "load_recipes": {"recipes": {"nemotron74_shared_kv_http_v1": {
            "model": model, "status": "validated", "procedure": "shared_kv_http_v1"}}},
        "subagent_catalog": {"schema_version": 1, "candidates": {
            "nemotron74_shared_1060": {"provider": "rig_shared_chat", "model": model,
                "agent_types": ["coding", "research"], "context_cap": 262144,
                "context_allocation": "dynamic_at_attach", "context_min": 16384,
                "context_default": 262144,
                "memory_estimate": {"fixed_compute_mib": 6, "kv_mib_at_262144": 2048,
                                    "admission_reserve_mib": 256},
                "max_attached_chats": 4}}}})
    rig = NemoRig()
    class Manager:
        pass
    manager = Manager()
    manager.rig = rig
    app = FastAPI()
    app.include_router(manager_router(manager, lambda: {}, "gpu-rig"))
    with TestClient(app) as client:
        found = client.get("/v1/manager/local-agent-capabilities").json()
        assert found["candidates"][0]["candidate_id"] == "nemotron74_shared_1060"
        assert found["candidates"][0] == {
            "candidate_id": "nemotron74_shared_1060", "agent_types": ["coding", "research"],
            "model": model, "status": "experimental", "context_length": 262144,
            "context_allocation": "dynamic_at_attach", "context_min": 16384,
            "context_default": 262144,
            "memory_estimate": {"fixed_compute_mib": 6, "kv_mib_at_262144": 2048,
                                "admission_reserve_mib": 256},
            "max_concurrent_children": 4,
            "loaded": True, "service_instance_id": "load-1", "active_reservations": [],
            "active_children": 0, "available_worker_slots": 4,
            "available_worker_gpu_ids": [1], "capacity_context_tokens": 262144}
        got = client.post("/v1/manager/agent-routes/resolve", json={
            "parent_route": "gpu-rig/shared-nemo-main", "agent_type": "coding"})
        assert got.status_code == 200, got.text
        child = got.json()
        assert child["route"].startswith("gpu-rig/shared-nemo-child-")
        assert child["gpu_ids"] == [1]
        assert child["service_instance_id"] == "load-1"
        assert child["reservation"]["context_capacity"] == 262144
        assert child["admission"]["mode"] == "shared_model"
        occupied = client.get("/v1/manager/local-agent-capabilities").json()["candidates"][0]
        assert occupied["active_children"] == 1
        assert occupied["available_worker_slots"] == 3
        assert occupied["active_reservations"][0]["route"] == child["route"]
        assert occupied["active_reservations"][0]["reservation_id"] == child["reservation"]["reservation_id"]
        released = client.post("/v1/manager/agent-routes/release", json={
            "parent_route": "gpu-rig/shared-nemo-main", "route": child["route"],
            "service_instance_id": child["service_instance_id"],
            "reservation_id": child["reservation"]["reservation_id"]})
        assert released.status_code == 200, released.text
        assert len(rig.chats) == 1
        repeated = client.post("/v1/manager/agent-routes/release", json={
            "parent_route": "gpu-rig/shared-nemo-main", "route": child["route"],
            "service_instance_id": child["service_instance_id"],
            "reservation_id": child["reservation"]["reservation_id"]})
        assert repeated.status_code == 200, repeated.text
        assert repeated.json()["already_released"] is True

def test_manager_requests_context_size_for_new_shared_child(monkeypatch):
    model = "qwen3.6:35b-a3b-xl"
    monkeypatch.setattr("rig_control.manager.routes.models.catalog", lambda: {
        "models": {model: {"status": "active"}},
        "load_recipes": {"recipes": {"qwen35_shared_kv_http_v1": {
            "model": model, "status": "validated", "procedure": "shared_kv_http_v1"}}},
        "subagent_catalog": {"schema_version": 1, "candidates": {
            "qwen35_shared_1060": {"provider": "rig_shared_chat", "model": model,
                "agent_types": ["coding"], "context_cap": 262144,
                "context_allocation": "dynamic_at_attach", "max_attached_chats": 5}}}})
    rig = SharedRig()
    class Manager:
        pass
    manager = Manager()
    manager.rig = rig
    app = FastAPI()
    app.include_router(manager_router(manager, lambda: {}, "gpu-rig"))
    with TestClient(app) as client:
        response = client.post("/v1/manager/agent-routes/resolve", json={
            "parent_route": "cloud/codex", "agent_type": "coding",
            "candidate_id": "qwen35_shared_1060", "min_context": 16000,
            "context_tokens": 32768})
        assert response.status_code == 200, response.text
        child = response.json()
        assert child["context_length"] == 32768
        assert child["reservation"]["context_capacity"] == 32768
        assert child["admission"]["served_context_tokens"] == 32768

def test_explicit_local_agent_model_swap_is_instance_checked(monkeypatch):
    qwen, nemo = "qwen3.6:35b-a3b-xl", "nemotron-3.5-bolt:74b-a9b"
    catalog = {"models": {qwen: {"status": "active"}, nemo: {"status": "active"}},
        "load_recipes": {"recipes": {"nemotron74_shared_kv_http_v1": {
            "model": nemo, "status": "validated", "procedure": "shared_kv_http_v1"}}},
        "subagent_catalog": {"candidates": {
            "qwen35_shared_1060": {"provider": "rig_shared_chat", "model": qwen,
                "requires": {"mode": "gpu0-main-q4"}, "agent_types": ["research"],
                "context_cap": 262144},
            "nemotron74_shared_1060": {"provider": "rig_shared_chat", "model": nemo,
                "requires": {"mode": "gpu0-main-f16-experts1234"},
                "agent_types": ["research"], "context_cap": 262144}}}}
    monkeypatch.setattr("rig_control.manager.routes.models.catalog", lambda: catalog)

    class SwapRig:
        def __init__(self):
            self.model, self.mode, self.instance = qwen, "gpu0-main-q4", "qwen-1"
            self.chats = [{"chat_id": "main", "gpu": 0, "context_cap": 262144}]
            self.calls = []

        async def status(self):
            return {"shared_kv_service": {"status": "ready", "model": self.model,
                    "mode": self.mode, "instance_id": self.instance,
                    "chats": list(self.chats)}}

        async def shared_kv(self, action, **kwargs):
            self.calls.append((action, kwargs))
            if action == "stop":
                self.model = self.mode = self.instance = None
                self.chats = []
            elif action == "start":
                self.model = kwargs["model"]
                self.mode = "gpu0-main-f16-experts1234" if self.model == nemo else "gpu0-main-q4"
                self.instance = "nemo-2" if self.model == nemo else "qwen-2"
                self.chats = [{"chat_id": "main", "gpu": 0, "context_cap": 262144}]
            return {"status": "ready"}

    rig = SwapRig()
    manager = type("Manager", (), {"rig": rig})()
    app = FastAPI()
    app.add_exception_handler(ManagerError,
        lambda _request, exc: JSONResponse(status_code=exc.status, content={"error": str(exc)}))
    app.include_router(manager_router(manager, lambda: {}, "gpu-rig"))
    with TestClient(app) as client:
        stale = client.post("/v1/manager/local-agent-model/swap", json={
            "candidate_id": "nemotron74_shared_1060",
            "expected_candidate_id": "qwen35_shared_1060", "expected_instance_id": "old"})
        assert stale.status_code == 409
        swapped = client.post("/v1/manager/local-agent-model/swap", json={
            "candidate_id": "nemotron74_shared_1060",
            "expected_candidate_id": "qwen35_shared_1060", "expected_instance_id": "qwen-1"})
        assert swapped.status_code == 200, swapped.text
        assert swapped.json() == {"swapped": True,
            "from_candidate_id": "qwen35_shared_1060",
            "to_candidate_id": "nemotron74_shared_1060",
            "service_instance_id": "nemo-2", "model": nemo,
            "mode": "gpu0-main-f16-experts1234"}
        assert [call[0] for call in rig.calls] == ["stop", "start"]

def test_local_agent_model_swap_refuses_active_child(monkeypatch):
    model = "qwen3.6:35b-a3b-xl"
    monkeypatch.setattr("rig_control.manager.routes.models.catalog", lambda: {
        "models": {model: {"status": "active"}}, "subagent_catalog": {"candidates": {
            "qwen35_shared_1060": {"provider": "rig_shared_chat", "model": model,
                "agent_types": ["research"], "context_cap": 262144}}}})
    rig = SharedRig()
    rig.chats.append({"chat_id": "child-busy", "gpu": 1, "context_cap": 32768})
    manager = type("Manager", (), {"rig": rig})()
    app = FastAPI()
    app.add_exception_handler(ManagerError,
        lambda _request, exc: JSONResponse(status_code=exc.status, content={"error": str(exc)}))
    app.include_router(manager_router(manager, lambda: {}, "gpu-rig"))
    with TestClient(app) as client:
        response = client.post("/v1/manager/local-agent-model/swap", json={
            "candidate_id": "qwen35_shared_1060",
            "expected_candidate_id": "qwen35_shared_1060", "expected_instance_id": "qwen-load-1"})
    assert response.status_code == 409
    assert "active local-agent children" in response.text

def test_manager_rolls_back_a_malformed_shared_admission(monkeypatch):
    model = "qwen3.6:35b-a3b-xl"
    monkeypatch.setattr("rig_control.manager.routes.models.catalog", lambda: {
        "models": {model: {"status": "active"}},
        "subagent_catalog": {"schema_version": 1, "candidates": {
            "qwen35_shared_1060": {"provider": "rig_shared_chat", "model": model,
                "agent_types": ["coding"], "context_cap": 262144,
                "context_allocation": "dynamic_at_attach", "context_default": 32768,
                "max_attached_chats": 5}}}})

    class MalformedRig(SharedRig):
        async def shared_kv(self, action, **kwargs):
            result = await super().shared_kv(action, **kwargs)
            if action == "attach":
                result["reservation"].pop("reservation_id")
            return result

    rig = MalformedRig()
    manager = type("Manager", (), {"rig": rig})()
    app = FastAPI()
    @app.exception_handler(ManagerError)
    async def manager_error(_request, exc):
        return JSONResponse(status_code=exc.status, content={"error": str(exc)})
    app.include_router(manager_router(manager, lambda: {}, "gpu-rig"))
    with TestClient(app) as client:
        response = client.post("/v1/manager/agent-routes/resolve", json={
            "parent_route": "cloud/codex", "agent_type": "coding",
            "worker_strategy": "shared_model"})
    assert response.status_code == 502, response.text
    assert [call[0] for call in rig.calls] == ["attach", "detach"]
    assert len(rig.chats) == 1

def test_manager_can_choose_a_separate_small_model_instead_of_loaded_shared_model(monkeypatch):
    shared_model = "qwen3.6:35b-a3b-xl"
    worker_model = "qwen2.5-coder:7b"
    monkeypatch.setattr("rig_control.manager.routes.models.catalog", lambda: {
        "models": {shared_model: {"status": "active"}, worker_model: {"status": "active"}},
        "roles": {"brain_code": {"model": worker_model}},
        "subagent_catalog": {"schema_version": 1, "candidates": {
            "qwen35_shared_1060": {"provider": "rig_shared_chat", "model": shared_model,
                "agent_types": ["coding"], "context_cap": 262144,
                "context_allocation": "dynamic_at_attach", "max_attached_chats": 5}}}})
    rig = SharedRig()
    class Manager:
        async def status(self):
            return {"profiles": {"coder": {"id": "coder", "model": worker_model,
                    "state": "ready", "gpus": [2], "ctx_size": 32768, "port": 11436}}}
    manager = Manager()
    manager.rig = rig
    app = FastAPI()
    app.include_router(manager_router(manager, lambda: {
        "coder": {"model": worker_model}}, "gpu-rig"))
    with TestClient(app) as client:
        response = client.post("/v1/manager/agent-routes/resolve", json={
            "parent_route": "cloud/codex", "agent_type": "coding",
            "worker_strategy": "separate_model", "min_context": 16384})
        assert response.status_code == 200, response.text
        child = response.json()
        assert child["route"] == "gpu-rig/coder"
        assert child["gpu_ids"] == [2]
        assert child["admission"] == {"mode": "separate_model",
                                      "served_context_tokens": 32768}
        assert rig.calls == []

def test_manager_can_select_a_benchmark_named_separate_candidate(monkeypatch):
    worker_model = "ministral-3:3b"
    monkeypatch.setattr("rig_control.manager.routes.models.catalog", lambda: {
        "models": {worker_model: {"status": "active"}}, "roles": {},
        "subagent_catalog": {"schema_version": 1, "candidates": {
            "ministral3_3b_separate": {"provider": "gateway_profile",
                "model": worker_model, "agent_types": ["coding", "research"],
                "status": "candidate"}}}})
    rig = SharedRig()

    class Manager:
        async def status(self):
            return {"profiles": {"small": {"id": "small", "model": worker_model,
                    "state": "ready", "gpus": [3], "ctx_size": 16384, "port": 11438}}}

    manager = Manager()
    manager.rig = rig
    app = FastAPI()
    app.include_router(manager_router(manager, lambda: {
        "small": {"model": worker_model}}, "gpu-rig"))
    async def check():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            capabilities = (await client.get("/v1/manager/local-agent-capabilities")).json()
            candidate = capabilities["candidates"][0]
            assert candidate["candidate_id"] == "ministral3_3b_separate"
            assert candidate["loaded"] is True and candidate["ready_routes"] == ["gpu-rig/small"]
            response = await client.post("/v1/manager/agent-routes/resolve", json={
                "parent_route": "cloud/codex", "agent_type": "coding",
                "worker_strategy": "separate_model", "candidate_id": "ministral3_3b_separate"})
            assert response.status_code == 200, response.text
            child = response.json()
            assert child["candidate_id"] == "ministral3_3b_separate"
            assert child["route"] == "gpu-rig/small" and child["gpu_ids"] == [3]

    asyncio.run(check())

    def loading(self):
        return []

def test_manager_attaches_and_releases_cataloged_child(monkeypatch):
    rig = SharedRig()
    class Manager:
        pass
    manager = Manager()
    manager.rig = rig
    monkeypatch.setattr("rig_control.manager.routes.models.catalog", lambda: {
        "subagent_catalog": {"schema_version": 1, "candidates": {
            "qwen35_shared_1060": {"provider": "rig_shared_chat",
                "model": "qwen3.6:35b-a3b-xl", "agent_types": ["coding"]}}}})
    app = FastAPI()
    app.include_router(manager_router(manager, lambda: {}, "gpu-rig"))
    with TestClient(app) as client:
        available = client.get("/v1/manager/local-agent-capabilities")
        assert available.status_code == 200
        assert available.json()["candidates"][0]["candidate_id"] == "qwen35_shared_1060"
        got = client.post("/v1/manager/agent-routes/resolve", json={
            "parent_route": "gpu-rig/shared-35b-main", "agent_type": "coding",
            "candidate_id": "qwen35_shared_1060"})
        assert got.status_code == 200, got.text
        child = got.json()
        assert child["ephemeral"] is True and child["gpu_ids"] == [1]
        assert child["route"].startswith("gpu-rig/shared-35b-child-")
        released = client.post("/v1/manager/agent-routes/release", json={
            "parent_route": "gpu-rig/shared-35b-main", "route": child["route"]})
        assert released.status_code == 200, released.text
        assert len(rig.chats) == 1

def test_headless_pool_is_mode_matched_and_can_serve_cloud_parent(monkeypatch):
    class HeadlessRig(SharedRig):
        def __init__(self):
            super().__init__()
            self.chats = []

        async def status(self):
            return {"profiles": [], "shared_kv_service": {
                "status": "ready", "model": "qwen3.6:35b-a3b-xl",
                "mode": "gpu0-worker-pool-q4", "instance_id": "qwen-load-1",
                "chats": [dict(row) for row in self.chats],
                "reservations": {"main": {"context_capacity": 0,
                    "placement": {"gpus": [0]}, "reservation_scope": "service_parent"}},
                "worker_capacity": {"status": "observed", "context_capacity": 65536,
                    "available_chat_slots": 32, "free_gpu_ids": [1, 2, 3, 4, 5]}}}

        async def shared_kv(self, action, **kwargs):
            result = await super().shared_kv(action, **kwargs)
            if action == "attach":
                result["reservation"]["placement"]["runtime_gpu"] = result["gpu"]
            return result

    catalog = {"subagent_catalog": {"schema_version": 1, "candidates": {
        "qwen35_shared_1060": {"provider": "rig_shared_chat",
            "model": "qwen3.6:35b-a3b-xl", "agent_types": ["coding"],
            "requires": {"mode": "gpu0-main-q4"}},
        "qwen35_headless_worker_pool": {"provider": "rig_shared_chat",
            "model": "qwen3.6:35b-a3b-xl", "agent_types": ["coding"],
            "context_cap": 262144, "context_default": 65536,
            "context_allocation": "dynamic_at_attach",
            "requires": {"mode": "gpu0-worker-pool-q4"}}}}}
    rig = HeadlessRig()
    class Manager:
        pass
    manager = Manager()
    manager.rig = rig
    monkeypatch.setattr("rig_control.manager.routes.models.catalog", lambda: catalog)
    app = FastAPI()
    app.include_router(manager_router(manager, lambda: {}, "gpu-rig"))
    with TestClient(app) as client:
        capabilities = client.get("/v1/manager/local-agent-capabilities").json()["candidates"]
        loaded = {row["candidate_id"]: row["loaded"] for row in capabilities}
        assert loaded == {"qwen35_shared_1060": False,
                          "qwen35_headless_worker_pool": True}
        got = client.post("/v1/manager/agent-routes/resolve", json={
            "parent_route": "cloud/codex", "agent_type": "coding",
            "candidate_id": "qwen35_headless_worker_pool"})
        assert got.status_code == 200, got.text
        assert got.json()["candidate_id"] == "qwen35_headless_worker_pool"
        assert got.json()["reservation"]["placement"]["runtime_gpu"] == 1
        child = got.json()
        released = client.post("/v1/manager/agent-routes/release", json={
            "parent_route": "cloud/codex", "route": child["route"],
            "service_instance_id": child["service_instance_id"],
            "reservation_id": child["reservation"]["reservation_id"]})
        assert released.status_code == 200, released.text
        assert released.json()["released"] == child["route"]
        assert rig.chats == []
        repeated = client.post("/v1/manager/agent-routes/release", json={
            "parent_route": "cloud/codex", "route": child["route"],
            "service_instance_id": child["service_instance_id"],
            "reservation_id": child["reservation"]["reservation_id"]})
        assert repeated.status_code == 200, repeated.text
        assert repeated.json()["already_released"] is True

def test_cloud_parent_starts_shared_service_and_releases_worker(monkeypatch):
    rig = SharedRig(ready=False)
    class Manager:
        pass
    manager = Manager()
    manager.rig = rig
    monkeypatch.setattr("rig_control.manager.routes.models.catalog", lambda: {
        "subagent_catalog": {"schema_version": 1, "candidates": {
            "qwen35_shared_1060": {"provider": "rig_shared_chat",
                "model": "qwen3.6:35b-a3b-xl", "agent_types": ["research"]}}}})
    app = FastAPI()
    app.include_router(manager_router(manager, lambda: {}, "gpu-rig"))
    with TestClient(app) as client:
        got = client.post("/v1/manager/agent-routes/resolve", json={
            "parent_route": "cloud/codex", "agent_type": "research",
            "candidate_id": "qwen35_shared_1060"})
        assert got.status_code == 200, got.text
        child = got.json()
        assert child["gpu_ids"] == [1] and child["parent_route"] == "cloud/codex"
        assert [call[0] for call in rig.calls] == ["start", "attach"]
        released = client.post("/v1/manager/agent-routes/release", json={
            "parent_route": "cloud/codex", "route": child["route"]})
        assert released.status_code == 200, released.text
        assert [call[0] for call in rig.calls] == ["start", "attach", "detach"]

def test_cloud_parent_default_uses_ready_shared_service(monkeypatch):
    rig = SharedRig()
    class Manager:
        pass
    manager = Manager()
    manager.rig = rig
    monkeypatch.setattr("rig_control.manager.routes.models.catalog", lambda: {
        "subagent_catalog": {"schema_version": 1, "candidates": {
            "qwen35_shared_1060": {"provider": "rig_shared_chat",
                "model": "qwen3.6:35b-a3b-xl", "agent_types": ["research"]}}}})
    app = FastAPI()
    app.include_router(manager_router(manager, lambda: {}, "gpu-rig"))
    with TestClient(app) as client:
        got = client.post("/v1/manager/agent-routes/resolve", json={
            "parent_route": "cloud/codex", "agent_type": "research"})
        assert got.status_code == 200, got.text
        assert got.json()["candidate_id"] == "qwen35_shared_1060"
        assert [call[0] for call in rig.calls] == ["attach"]
        released = client.post("/v1/manager/agent-routes/release", json={
            "parent_route": "cloud/codex", "route": got.json()["route"]})
        assert released.status_code == 200
