"""HTTP API of the rig manager, mounted at /v1/manager (docs/MANAGER.md, "The client process")."""

from __future__ import annotations

import asyncio
import json
import uuid
from typing import Any, Callable

from fastapi import APIRouter, Request

from . import campaigns, fit, models, observations
from .core import Manager
from .errors import ManagerError
from ..rig import RigUnavailable
from ..shared_chat_contract import ROUTE_PREFIX_BY_MODEL, serving_approved, validate_admission


# Agent capability policy names reviewed benchmarking roles, never models or GPUs.
# Ready routes favor quality; cold loads try smaller workers first.
AGENT_ROLE_ORDER = {
    "coding": ("brain_code", "single_1060_code"),
    "research": ("brain", "single_1060", "text_extraction"),
}
def _shared_candidates(catalog: dict) -> dict[str, dict]:
    """Reviewed shared services; a candidate only attaches to its already-loaded model."""
    entries = (catalog.get("subagent_catalog") or {}).get("candidates") or {}
    return {cid: {**entry, "route_prefix": ROUTE_PREFIX_BY_MODEL[entry["model"]]}
            for cid, entry in entries.items()
            if entry.get("provider") == "rig_shared_chat"
            and entry.get("model") in ROUTE_PREFIX_BY_MODEL
            and serving_approved(catalog, entry["model"])}


def _profile_candidates(catalog: dict, published: dict[str, dict]) -> dict[str, dict]:
    """Benchmark-reviewed separate children that Gateway can publish by model identity."""
    entries = (catalog.get("subagent_catalog") or {}).get("candidates") or {}
    models = catalog.get("models") or {}
    published_models = {row.get("model") for row in published.values()}
    return {cid: entry for cid, entry in entries.items()
            if entry.get("provider") == "gateway_profile"
            and entry.get("model") in published_models
            and (models.get(entry.get("model")) or {}).get("status") == "active"
            and not (models.get(entry.get("model")) or {}).get("in_testing")}


def _candidate_profile_order(entry: dict, published: dict[str, dict]) -> list[str]:
    return [pid for pid, profile in published.items() if profile.get("model") == entry.get("model")]


def _candidate_loaded(service: dict, entry: dict) -> bool:
    required = entry.get("requires") or {}
    return (service.get("status") == "ready"
            and service.get("model") == entry.get("model")
            and (not required.get("mode") or service.get("mode") == required["mode"]))


def _agent_profile_order(agent_type: str, published: dict[str, dict], *, cold: bool = False) -> list[str]:
    catalog = models.catalog()
    entries = catalog.get("models", {})
    roles = catalog.get("roles", {})
    order = list(AGENT_ROLE_ORDER[agent_type])
    if cold:
        order.reverse()
    selected = []
    for role_name in order:
        model = (roles.get(role_name) or {}).get("model")
        entry = entries.get(model, {})
        if entry.get("status") != "active" or entry.get("in_testing"):
            continue
        for pid, profile in published.items():
            if profile.get("model") == model and pid not in selected:
                selected.append(pid)
    return selected


async def _body(request: Request) -> dict[str, Any]:
    raw = await request.body()
    try:
        body = json.loads(raw or b"{}")
    except ValueError as exc:
        raise ManagerError(400, f"invalid JSON body: {exc}") from exc
    if not isinstance(body, dict):
        raise ManagerError(400, "body must be a JSON object")
    return body


def _require(body: dict, *keys: str) -> None:
    missing = [k for k in keys if not body.get(k)]
    if missing:
        raise ManagerError(400, f"missing field(s): {', '.join(missing)}")


def router(manager: Manager, profiles: Callable[[], dict[str, dict]], model_prefix: str = "gpu-rig") -> APIRouter:
    r = APIRouter(prefix="/v1/manager")

    @r.get("/status")
    async def status():
        return await manager.status()

    @r.get("/model-offers")
    async def model_offers():
        """Reviewed capabilities for runtime clients; benchmark candidates stay private."""
        catalog = models.catalog()
        available = catalog.get("models", {})
        data = []
        for capability, role in sorted(catalog.get("roles", {}).items()):
            model = role.get("model")
            entry = available.get(model, {})
            if entry.get("status") != "active" or entry.get("in_testing"):
                continue
            data.append({"capability": capability, "model": model,
                         "profile": role.get("profile") or "default",
                         "good_for": role.get("good_for") or [],
                         "description": role.get("why") or ""})
        return {"object": "model_offers", "version": 1, "data": data}

    @r.get("/chat/availability")
    async def chat_availability(profile_id: str | None = None):
        if profile_id is not None and profile_id not in profiles():
            raise ManagerError(404, "unknown published profile")
        return await manager.chat_availability(profile_id)

    @r.get("/subagent-candidates")
    async def subagent_candidates():
        """Imported model-library identities; readiness is checked by each provider at use time."""
        full_catalog = models.catalog()
        catalog = full_catalog.get("subagent_catalog", {"schema_version": 1, "candidates": {}})
        allowed = {**_shared_candidates(full_catalog),
                   **_profile_candidates(full_catalog, profiles())}
        return {**catalog, "candidates": {cid: row for cid, row in (catalog.get("candidates") or {}).items()
                if row.get("provider") not in ("rig_shared_chat", "gateway_profile") or cid in allowed}}

    @r.get("/local-agent-capabilities")
    async def local_agent_capabilities():
        """Cloud-child choices plus truthful fixed-context and live slot information."""
        full_catalog = models.catalog()
        try:
            service = (await manager.rig.status()).get("shared_kv_service") or {}
        except RigUnavailable:
            service = {}
        published = profiles()
        separate_candidates = _profile_candidates(full_catalog, published)
        loaded_profiles = {}
        if separate_candidates:
            try:
                observed = await manager.status()
                loaded_profiles = observed.get("profiles") or {}
                if isinstance(loaded_profiles, list):
                    loaded_profiles = {row.get("id"): row for row in loaded_profiles if isinstance(row, dict)}
            except (ManagerError, RigUnavailable):
                loaded_profiles = {}
        candidates = []
        for candidate_id, entry in _shared_candidates(full_catalog).items():
            loaded = _candidate_loaded(service, entry)
            active = sum(isinstance(chat, dict) and
                         str(chat.get("chat_id", "")).startswith("child-")
                         for chat in (service.get("chats") or [])) if loaded else 0
            limit = entry.get("max_attached_chats")
            live_reservations = [{**chat["reservation"],
                                  "route": model_prefix + "/" + entry["route_prefix"] + chat["chat_id"],
                                  "chat_id": chat["chat_id"]}
                                 for chat in (service.get("chats") or [])
                                 if loaded and isinstance(chat, dict)
                                 and isinstance(chat.get("chat_id"), str)
                                 and chat["chat_id"].startswith("child-")
                                 and isinstance(chat.get("reservation"), dict)]
            worker_capacity = service.get("worker_capacity") or {}
            authoritative = loaded and worker_capacity.get("status") == "observed"
            candidates.append({"candidate_id": candidate_id,
                               "agent_types": entry.get("agent_types", []),
                               "model": entry.get("model"),
                               "status": entry.get("status", "experimental"),
                               "context_length": entry.get("context_cap"),
                               "context_allocation": entry.get("context_allocation", "fixed_per_chat"),
                               **({"context_min": entry["context_min"]}
                                  if type(entry.get("context_min")) is int else {}),
                               **({"context_default": entry["context_default"]}
                                  if type(entry.get("context_default")) is int else {}),
                               **({"memory_estimate": entry["memory_estimate"]}
                                  if isinstance(entry.get("memory_estimate"), dict) else {}),
                               "max_concurrent_children": limit,
                               "loaded": loaded,
                               "service_instance_id": service.get("instance_id") if loaded else None,
                               "active_reservations": live_reservations,
                               "active_children": active,
                               "available_worker_slots": (
                                   worker_capacity.get("available_chat_slots",
                                                       worker_capacity.get("free_gpu_count"))
                                   if authoritative else None),
                               "available_worker_gpu_ids": worker_capacity.get("free_gpu_ids")
                                   if authoritative else None,
                               "capacity_context_tokens": worker_capacity.get("context_capacity")
                                   if authoritative else None})
        for candidate_id, entry in separate_candidates.items():
            profile_ids = _candidate_profile_order(entry, published)
            ready = [loaded_profiles[pid] for pid in profile_ids
                     if pid in loaded_profiles and loaded_profiles[pid].get("state") == "ready"
                     and loaded_profiles[pid].get("model") == entry["model"]]
            candidates.append({"candidate_id": candidate_id,
                               "provider": "gateway_profile",
                               "agent_types": entry.get("agent_types", []),
                               "model": entry["model"],
                               "status": entry.get("status", "candidate"),
                               "profile_ids": profile_ids,
                               "loaded": bool(ready),
                               "ready_routes": [model_prefix + "/" + row["id"] for row in ready],
                               "gpu_ids": sorted({gpu for row in ready for gpu in (row.get("gpus") or [])}),
                               "context_length": max((row.get("ctx_size") or 0 for row in ready), default=None)})
        return {"object": "local_agent_capabilities", "version": 1,
                "roles": [{"agent_type": role, "description": (
                    "Analyze sources, extract evidence, and identify limits" if role == "research" else
                    "Review or solve bounded coding tasks")}
                    for role in AGENT_ROLE_ORDER],
                "default": "The manager selects a ready local model or loads one for the requested role; slot admission is authoritative at delegation time",
                "candidates": candidates}

    @r.post("/local-agent-model/swap")
    async def swap_local_agent_model(request: Request):
        """Explicitly replace one observed shared local-agent service with another."""
        body = await _body(request)
        target_id = body.get("candidate_id")
        expected_id = body.get("expected_candidate_id")
        expected_instance = body.get("expected_instance_id")
        if (not isinstance(target_id, str) or not target_id
                or (expected_id is not None and not isinstance(expected_id, str))
                or (expected_instance is not None and not isinstance(expected_instance, str))):
            raise ManagerError(400, "candidate_id and optional expected candidate/instance strings are required")
        shared = _shared_candidates(models.catalog())
        target = shared.get(target_id)
        if target is None:
            raise ManagerError(400, "target is not a reviewed shared local-agent candidate")
        service = (await manager.rig.status()).get("shared_kv_service") or {}
        current_id = next((cid for cid, entry in shared.items()
                           if _candidate_loaded(service, entry)), None)
        if service.get("status") == "ready" and current_id is None:
            raise ManagerError(409, "loaded shared service is not a reviewed local-agent candidate")
        if current_id != expected_id:
            raise ManagerError(409, "loaded local-agent candidate changed before swap")
        if current_id is not None and service.get("instance_id") != expected_instance:
            raise ManagerError(409, "loaded local-agent service instance changed before swap")
        children = [row.get("chat_id") for row in (service.get("chats") or [])
                    if isinstance(row, dict) and str(row.get("chat_id", "")).startswith("child-")]
        if children:
            raise ManagerError(409, "wait for active local-agent children before swapping models")
        if current_id == target_id:
            return {"swapped": False, "from_candidate_id": current_id,
                    "to_candidate_id": target_id, "service_instance_id": service.get("instance_id")}
        if current_id is not None:
            await manager.rig.shared_kv("stop", model=service["model"])
        start = {"model": target["model"]}
        if (target.get("requires") or {}).get("mode") == "gpu0-worker-pool-q4":
            start["main_context_cap"] = 0
        await manager.rig.shared_kv("start", **start)
        observed = (await manager.rig.status()).get("shared_kv_service") or {}
        if not _candidate_loaded(observed, target):
            raise ManagerError(502, "requested local-agent model did not become ready")
        return {"swapped": True, "from_candidate_id": current_id,
                "to_candidate_id": target_id, "service_instance_id": observed.get("instance_id"),
                "model": observed.get("model"), "mode": observed.get("mode")}

    @r.post("/agent-routes/release")
    async def release_agent_route(request: Request):
        body = await _body(request)
        route = body.get("route")
        parent = body.get("parent_route")
        expected_instance = body.get("service_instance_id")
        expected_reservation = body.get("reservation_id")
        if expected_instance is not None and not isinstance(expected_instance, str):
            raise ManagerError(400, "service_instance_id must be a string")
        if expected_reservation is not None and not isinstance(expected_reservation, str):
            raise ManagerError(400, "reservation_id must be a string")
        shared = {model_prefix + "/" + route_prefix: model
                  for model, route_prefix in ROUTE_PREFIX_BY_MODEL.items()}
        prefix = next((p for p in shared if isinstance(route, str) and route.startswith(p + "child-")), None)
        if prefix is None or not isinstance(parent, str) or \
                (parent != "cloud/codex" and not parent.startswith(prefix)):
            raise ManagerError(400, "an attached shared child route and parent are required")
        child_id = route[len(prefix):]
        status = await manager.rig.status()
        service = status.get("shared_kv_service") or {}
        chats = {row.get("chat_id"): row for row in service.get("chats", []) if isinstance(row, dict)}
        parent_chat = "main" if parent == "cloud/codex" else parent[len(prefix):]
        if service.get("status") != "ready":
            raise ManagerError(409, "shared parent is not ready")
        if parent_chat in chats:
            parent_gpu = chats[parent_chat].get("gpu")
        elif parent == "cloud/codex":
            # A headless shared service reserves its parent weights without a
            # main chat. Its zero-context service reservation is the parent.
            main = (service.get("reservations") or {}).get("main") or {}
            parent_gpus = (main.get("placement") or {}).get("gpus") or []
            if main.get("context_capacity") != 0 or not parent_gpus or type(parent_gpus[0]) is not int:
                raise ManagerError(409, "shared parent is not ready")
            parent_gpu = parent_gpus[0]
        else:
            raise ManagerError(409, "shared parent is not ready")
        if expected_instance is not None and service.get("instance_id") != expected_instance:
            raise ManagerError(409, "shared service instance changed before release")
        if child_id not in chats:
            if expected_instance is None:
                raise ManagerError(409, "shared child is not ready")
            return {"released": route, "already_released": True,
                    "service_instance_id": service.get("instance_id")}
        if chats[child_id].get("gpu") == parent_gpu:
            raise ManagerError(409, "child is not isolated from parent")
        if service.get("model") != shared[prefix]:
            raise ManagerError(409, "shared child model changed")
        reservation = chats[child_id].get("reservation")
        if expected_reservation is not None and (not isinstance(reservation, dict)
                or reservation.get("reservation_id") != expected_reservation):
            raise ManagerError(409, "shared child reservation changed before release")
        detached = await manager.rig.shared_kv("detach", model=shared[prefix], chat_id=child_id)
        return {"released": route,
                "service_instance_id": service.get("instance_id"),
                **({"released_reservation": detached["released_reservation"]}
                   if isinstance(detached.get("released_reservation"), dict) else {})}

    @r.post("/agent-routes/resolve")
    async def resolve_agent_route(request: Request):
        """Offer an already-ready, disjoint route for a child agent capability."""
        b = await _body(request)
        agent_type, parent_id = b.get("agent_type"), b.get("parent_route")
        candidate_id = b.get("candidate_id")
        worker_strategy = b.get("worker_strategy", "auto")
        if agent_type not in AGENT_ROLE_ORDER or not isinstance(parent_id, str):
            raise ManagerError(400, f"agent_type must be one of {sorted(AGENT_ROLE_ORDER)} and parent_route is required")
        if worker_strategy not in ("auto", "shared_model", "separate_model"):
            raise ManagerError(400, "worker_strategy must be auto, shared_model, or separate_model")
        prefix = model_prefix + "/"
        cloud_parent = parent_id == "cloud/codex"
        if not cloud_parent and not parent_id.startswith(prefix):
            raise ManagerError(400, "parent_route must be a gateway route or cloud/codex")
        min_context = b.get("min_context", 4096)
        context_tokens = b.get("context_tokens")
        if type(min_context) is not int or min_context < 1:
            raise ManagerError(400, "min_context must be a positive integer")
        if context_tokens is not None and (type(context_tokens) is not int
                or not 16384 <= context_tokens <= 262144):
            raise ManagerError(400, "context_tokens must be an integer from 16384 to 262144")
        if context_tokens is not None and context_tokens < min_context:
            raise ManagerError(400, "context_tokens cannot be smaller than min_context")
        catalog = models.catalog()
        shared = _shared_candidates(catalog)
        published = profiles()
        separate = _profile_candidates(catalog, published)
        profile_candidate = separate.get(candidate_id) if candidate_id is not None else None
        if candidate_id is not None:
            if worker_strategy == "separate_model" and candidate_id in shared:
                raise ManagerError(400, "shared-model candidate cannot use separate_model")
            if worker_strategy == "shared_model" and candidate_id in separate:
                raise ManagerError(400, "separate-model candidate cannot use shared_model")
            if candidate_id not in shared and candidate_id not in separate:
                raise ManagerError(400, "unknown or unavailable local candidate")
        service = (await manager.rig.status()).get("shared_kv_service") or {}
        selected = None
        inferred_shared = False
        if worker_strategy != "separate_model" and cloud_parent:
            if candidate_id is not None and profile_candidate is None:
                selected = candidate_id if candidate_id in shared else None
            elif service.get("status") == "ready":
                selected = next((cid for cid, row in shared.items()
                                 if _candidate_loaded(service, row)), None)
                inferred_shared = selected is not None
        elif worker_strategy != "separate_model":
            selected = next((cid for cid, row in shared.items()
                             if parent_id.startswith(prefix + row["route_prefix"])
                             and _candidate_loaded(service, row)), None)
            if selected and candidate_id not in (None, selected):
                raise ManagerError(400, "candidate is incompatible with the loaded shared model")
        if selected:
            entry = shared[selected]
            if agent_type not in entry.get("agent_types", []):
                raise ManagerError(409, "shared-chat subagent does not support this role")
            model = entry["model"]
            shared_prefix = prefix + entry["route_prefix"]
            model_arg = {"model": model}
            if cloud_parent and service.get("status") != "ready":
                required_mode = (entry.get("requires") or {}).get("mode")
                await manager.rig.shared_kv(
                    "start", **model_arg,
                    **({"main_context_cap": 0} if required_mode == "gpu0-worker-pool-q4" else {}))
                service = (await manager.rig.status()).get("shared_kv_service") or {}
            if not _candidate_loaded(service, entry):
                raise ManagerError(409, "candidate requires a different loaded shared-model mode")
            chats = {row.get("chat_id"): row for row in service.get("chats", []) if isinstance(row, dict)}
            headless = ((entry.get("requires") or {}).get("mode") == "gpu0-worker-pool-q4")
            parent = None if cloud_parent and headless else chats.get(
                "main" if cloud_parent else parent_id[len(shared_prefix):])
            if parent is not None:
                if type(parent.get("gpu")) is not int:
                    raise ManagerError(409, "parent shared chat has no GPU placement")
                if parent.get("context_cap", 0) < min_context:
                    raise ManagerError(409, "parent shared chat has insufficient context")
                parent_gpu = parent["gpu"]
            elif cloud_parent and headless:
                parent_gpus = (((service.get("reservations") or {}).get("main") or {})
                               .get("placement", {}).get("gpus") or [0])
                if type(parent_gpus[0]) is not int:
                    raise ManagerError(409, "headless shared service has no parent GPU placement")
                parent_gpu = parent_gpus[0]
            else:
                raise ManagerError(409, "parent shared chat is not ready")
            child_id = "child-" + uuid.uuid4().hex[:16]
            try:
                attach_args = {**model_arg, "chat_id": child_id}
                if context_tokens is not None:
                    attach_args["context_tokens"] = context_tokens
                elif entry.get("context_allocation") == "dynamic_at_attach":
                    attach_args["context_tokens"] = entry.get("context_default", entry["context_cap"])
                attached = await manager.rig.shared_kv("attach", **attach_args)
            except RigUnavailable as exc:
                if ("no eligible free GPU" in str(exc)
                        or "no admissible placement" in str(exc)):
                    if worker_strategy == "auto" and inferred_shared and candidate_id is None:
                        selected = None
                    else:
                        raise ManagerError(409, f"{selected} has no available worker slot") from exc
                if not inferred_shared:
                    raise
                candidate_id = None
            else:
                try:
                    admission = validate_admission(
                        attached, expected_instance=service.get("instance_id"),
                        parent_gpu=parent_gpu)
                except ValueError as exc:
                    try:
                        await manager.rig.shared_kv("detach", model=model, chat_id=child_id)
                    except Exception as rollback_exc:
                        raise ManagerError(502, f"invalid shared admission and rollback failed: {rollback_exc}") from exc
                    raise ManagerError(502, f"invalid shared admission: {exc}") from exc
                gpu = admission["gpu"]
                reservation = admission["reservation"]
                instance = admission["service_instance_id"]
                return {"agent_type": agent_type, "candidate_id": selected,
                        "route": shared_prefix + child_id, "model": model,
                        "gpu_ids": [gpu], "context_length": admission["context_capacity"],
                        "state": "ready", "parent_route": parent_id, "ephemeral": True,
                        "service_instance_id": instance, "reservation": reservation,
                        "admission": {"mode": "shared_model",
                                      "service_instance_id": instance,
                                      "reservation_id": reservation["reservation_id"],
                                      "requested_context_tokens": context_tokens,
                                      "served_context_tokens": admission["context_capacity"],
                                      "reservation": reservation}}
        if worker_strategy == "shared_model" and selected is None:
            raise ManagerError(409, "no requested shared-model worker is currently available")
        if candidate_id is not None and profile_candidate is None:
            raise ManagerError(400, "candidate is not available for this parent route")
        observed = await manager.status()
        if observed.get("from_copy"):
            raise ManagerError(503, "rig observation is stale")
        loaded = observed.get("profiles") or {}
        if isinstance(loaded, list):
            loaded = {row.get("id"): row for row in loaded if isinstance(row, dict)}
        parent = None if cloud_parent else loaded.get(parent_id[len(prefix):])
        if not cloud_parent and (not parent or parent.get("state") != "ready" or not parent.get("gpus")):
            raise ManagerError(409, "parent route is not ready with known GPU placement")
        parent_gpus = set() if cloud_parent else set(parent["gpus"])
        if profile_candidate is not None and agent_type not in profile_candidate.get("agent_types", []):
            raise ManagerError(409, "separate-model candidate does not support this role")
        profile_order = (_candidate_profile_order(profile_candidate, published)
                         if profile_candidate is not None else _agent_profile_order(agent_type, published))
        for pid in profile_order:
            row, spec = loaded.get(pid), published.get(pid)
            if not row or not spec or row.get("state") != "ready" or row.get("model") != spec.get("model"):
                continue
            gpus, context = row.get("gpus"), row.get("ctx_size")
            if (not gpus or not parent_gpus.isdisjoint(gpus) or
                    type(context) is not int or context < min_context or not row.get("port")):
                continue
            return {"agent_type": agent_type,
                    **({"candidate_id": candidate_id} if profile_candidate is not None else {}),
                    "route": prefix + pid, "model": row["model"],
                    "gpu_ids": sorted(gpus), "context_length": context, "state": "ready",
                    "admission": {"mode": "separate_model", "served_context_tokens": context}}
        # No ready offer: let the manager fit, lease, and load a published profile.
        # Existing ready routes on the parent's cards are never moved by this request.
        async def prepare_cold():
            failures = []
            cold_order = (_candidate_profile_order(profile_candidate, published)
                          if profile_candidate is not None else _agent_profile_order(agent_type, published, cold=True))
            for pid in cold_order:
                spec = published.get(pid)
                if not spec or pid in loaded:
                    continue
                try:
                    await manager.rig.ensure_ready(pid, start=lambda pid=pid, spec=spec: manager.load_profile(pid, spec))
                except (ManagerError, RigUnavailable) as exc:
                    if exc.status not in (409, 422, 503):
                        raise
                    failures.append(f"{pid}: {exc}")
                    continue
                refreshed = await manager.status()
                if refreshed.get("from_copy"):
                    raise ManagerError(503, "rig observation is stale after child load")
                rows = refreshed.get("profiles") or {}
                if isinstance(rows, list):
                    rows = {row.get("id"): row for row in rows if isinstance(row, dict)}
                parent_now, row = (None if cloud_parent else rows.get(parent_id[len(prefix):])), rows.get(pid)
                if ((cloud_parent or (parent_now and parent_now.get("state") == "ready"))
                        and row and row.get("state") == "ready"
                        and row.get("model") == spec.get("model") and row.get("gpus")
                        and (cloud_parent or set(parent_now.get("gpus") or []).isdisjoint(row["gpus"]))
                        and type(row.get("ctx_size")) is int and row["ctx_size"] >= min_context):
                    return {"agent_type": agent_type,
                            **({"candidate_id": candidate_id} if profile_candidate is not None else {}),
                            "route": prefix + pid, "model": row["model"],
                            "gpu_ids": sorted(row["gpus"]), "context_length": row["ctx_size"], "state": "ready",
                            "admission": {"mode": "separate_model",
                                          "served_context_tokens": row["ctx_size"]}}
                failures.append(f"{pid}: loaded route did not satisfy child offer")
            raise ManagerError(503, f"no {agent_type} route on GPUs separate from {parent_id}", failures=failures)
        try:
            return await asyncio.wait_for(prepare_cold(), timeout=300)
        except TimeoutError as exc:
            raise ManagerError(504, f"{agent_type} route was not ready within 300 seconds") from exc

    @r.get("/leases")
    async def leases():
        return {"leases": await manager.leases(), "claims": await manager.claims()}

    @r.post("/leases/preview")
    async def preview_lease(request: Request):
        """Nonbinding rig admission check; acquire rechecks under the rig lock."""
        b = await _body(request)
        return await manager.preview_lease(gpus_needed=b.get("gpus_needed"), gpu_ids=b.get("gpu_ids"),
                                           claim=b.get("claim"))

    @r.post("/leases", status_code=201)
    async def acquire(request: Request):
        b = await _body(request)
        _require(b, "controller", "owner", "run_id")
        lease = await manager.acquire(
            controller=b["controller"], owner=b["owner"], run_id=b["run_id"],
            gpus_needed=b.get("gpus_needed"), gpu_ids=b.get("gpu_ids"), claim=b.get("claim"),
            ttl_s=int(b.get("ttl_s", 120)), model=b.get("model", ""), purpose=b.get("purpose", ""),
            lane=b.get("lane"))
        return {"lease": lease, "port": lease["metadata"]["port"]}

    @r.post("/leases/{lease_id}/renew")
    async def renew(lease_id: str, request: Request):
        b = await _body(request)
        _require(b, "owner")
        return {"lease": await manager.renew(lease_id, b["owner"], int(b.get("ttl_s", 120)))}

    @r.post("/leases/{lease_id}/release")
    async def release(lease_id: str, request: Request):
        b = await _body(request)
        if not b.get("force"):
            _require(b, "owner")
        await manager.release(lease_id, b.get("owner", ""), force=bool(b.get("force")))
        return {"released": lease_id}

    @r.get("/claims")
    async def claims():
        return {"claims": await manager.claims()}

    @r.post("/claims", status_code=201)
    async def add_claim(request: Request):
        """Take cards out of the pool, or reserve them while they are still leased (a split waiting for
        its cards). ttl_s: the claim lapses unless its owner renews it."""
        b = await _body(request)
        _require(b, "name", "gpu_ids")
        return {"claim": await manager.add_claim(
            b["name"], b["gpu_ids"], b.get("note", ""), b.get("by", ""), owner=b.get("owner", ""),
            ttl_s=int(b["ttl_s"]) if b.get("ttl_s") else None, purpose=b.get("purpose", ""),
            model=b.get("model", ""))}

    @r.post("/claims/{name}/extend")
    async def extend_claim(name: str, request: Request):
        b = await _body(request)
        _require(b, "gpu_ids", "owner")
        return {"claim": await manager.extend_claim(name, [int(g) for g in b["gpu_ids"]], b["owner"])}

    @r.post("/claims/{name}/renew")
    async def renew_claim(name: str, request: Request):
        b = await _body(request)
        return {"claim": await manager.renew_claim(name, b.get("owner", ""),
                                                   int(b["ttl_s"]) if b.get("ttl_s") else None)}

    @r.post("/claims/{name}/release")
    async def remove_claim(name: str):
        return {"released": await manager.remove_claim(name)}

    @r.post("/kill")
    async def kill(request: Request):
        """Operator override: stop every runtime on these cards, release their leases, drop their claims."""
        b = await _body(request)
        _require(b, "gpu_ids")
        return await manager.kill([int(g) for g in b["gpu_ids"]], b.get("reason", ""))

    @r.post("/profiles/{profile_id}/load")
    async def load(profile_id: str, request: Request):
        published = profiles()
        if profile_id not in published:
            raise ManagerError(404, f"unknown profile {profile_id}; published: {sorted(published)}")
        b = await _body(request)
        gpu_ids = b.get("gpu_ids")
        if gpu_ids is not None and (not isinstance(gpu_ids, list) or not gpu_ids
                                    or any(type(g) is not int or g < 0 for g in gpu_ids)
                                    or len(set(gpu_ids)) != len(gpu_ids)):
            raise ManagerError(400, "gpu_ids must be a nonempty list of distinct nonnegative GPU indexes")
        port = await manager.rig.ensure_ready(
            profile_id, start=lambda: manager.load_profile(profile_id, published[profile_id], b.get("claim"), gpu_ids))
        if gpu_ids is not None:
            loaded = (await manager.observe(fresh=True)).get("profiles", {}).get(profile_id)
            if not loaded or sorted(loaded.get("gpus") or []) != sorted(gpu_ids):
                raise ManagerError(409, f"profile {profile_id} is ready on a different GPU set")
        return {"profile": profile_id, "port": port, "state": "ready"}

    @r.post("/profiles/{profile_id}/unload")
    async def unload(profile_id: str):
        return await manager.unload_profile(profile_id)

    def _model_request(b: dict) -> dict:
        _require(b, "model")
        ctx = b.get("ctx")
        return {"cards": int(b["cards"]) if b.get("cards") else None, "gpu_ids": b.get("gpu_ids"),
                "ctx": None if ctx in (None, "", "auto") else int(ctx),
                "ctx_min": int(b.get("ctx_min") or fit.CTX_MIN_DEFAULT), "use": b.get("use", "benchmark"),
                "profile": b.get("profile"), "claim": b.get("claim"),
                "placement_policy": b.get("placement_policy"),
                "margin_mb": int(b["margin_mb"]) if b.get("margin_mb") is not None else fit.MARGIN_MB_DEFAULT,
                "batch_size": int(b["batch_size"]) if b.get("batch_size") else None,
                "context_mode": b.get("context_mode", "working"),
                "engine": b.get("engine", "auto")}

    @r.post("/models/fit")
    async def fit_model(request: Request):
        """Would `model` fit on `cards` cards (or `gpu_ids`), and at what context? Nothing is loaded."""
        b = await _body(request)
        req = _model_request(b)
        return await manager.fit_model(b["model"], **req)

    @r.post("/models/load")
    async def load_model(request: Request):
        """Fit, lease, load on the rig, wait until it serves. Unload with /profiles/<profile>/unload.
        wait_s: reserve busy cards and wait for them (MANAGER.md, "Reservations"); force: kill what runs
        on the named cards first (operator override)."""
        b = await _body(request)
        req = _model_request(b)
        return await manager.load_model(b["model"], extra_args=[str(a) for a in b.get("extra_args") or []],
                                        wait_s=float(b.get("wait_s") or 0), force=bool(b.get("force")),
                                        owner=str(b.get("owner") or ""), **req)

    @r.post("/benchmark-campaigns/preview")
    async def preview_benchmark_campaign(request: Request):
        """Simulate a campaign over Orchestration-owned placement alternatives; change no rig state."""
        body = await _body(request)
        try:
            campaign, workloads = campaigns.compile_campaign(body)
            offered = await manager.schedule_preview(workloads, campaign["candidate_limit"])
            return campaigns.simulate(campaign, offered)
        except ValueError as exc:
            raise ManagerError(400, str(exc)) from exc

    @r.get("/observations")
    async def list_observations(kind: str = "load", model: str | None = None, limit: int = 200):
        """Every load the manager knows of (observations.py): its log in benchmarking plus, for `load`, the
        loads in the capacity records, newest last; `limit` 0 for all."""
        got = [r for r in observations.history(kind) if model is None or r.get("model") == model]
        return {"path": observations.path(), "count": len(got), "rows": got[-limit:] if limit else got}

    @r.get("/runs")
    async def runs():
        return {"runs": await manager.runs()}

    @r.post("/runs", status_code=201)
    async def start_run(request: Request):
        """Hand the rig a long run (campaign, lane) as a job in the shared report: {run, argv, owner?, cwd?,
        env?, resume_argv?, auto_resume?, needs? [{card_mb, count}], note?}."""
        b = await _body(request)
        _require(b, "run", "argv")
        return await manager.start_run(str(b["run"]), [str(a) for a in b["argv"]], owner=str(b.get("owner") or ""),
                                       cwd=b.get("cwd"), env=b.get("env") or {},
                                       resume_argv=[str(a) for a in b["resume_argv"]] if b.get("resume_argv") else None,
                                       auto_resume=bool(b.get("auto_resume")), needs=b.get("needs") or [],
                                       note=str(b.get("note") or ""))

    @r.post("/runs/{run}/recover")
    async def recover_run(run: str):
        return await manager.recover_run(run)

    @r.post("/runs/{run}/stop")
    async def stop_run(run: str, request: Request):
        b = await _body(request)
        return await manager.stop_run(run, str(b.get("reason") or ""))

    @r.post("/runs/{run}/dismiss")
    async def dismiss_run(run: str, request: Request):
        b = await _body(request)
        return await manager.dismiss_run(run, str(b.get("reason") or ""))

    @r.post("/owners/{who}/cancel")
    async def cancel_owner(who: str, request: Request):
        """Cancel everything one owner has on the rig: runs, queued items, loads, leases, claims, waits."""
        b = await _body(request)
        return await manager.cancel_owner(who, str(b.get("reason") or ""))

    @r.post("/reconcile")
    async def reconcile():
        return await manager.reconcile_once()

    @r.get("/controller")
    async def controller():
        await manager.observe(fresh=True)
        return manager.control()

    @r.post("/controller/claim")
    async def claim_control(request: Request):
        """Take control: when there is none or it is stale; {"force": true} is a handoff."""
        b = await _body(request)
        taken = await manager.take_control(force=bool(b.get("force")))
        await manager.observe(fresh=True)
        return {"taken": taken, **manager.control()}

    @r.post("/controller/release")
    async def release_control():
        await manager.release_control()
        return manager.control()

    @r.get("/plan")
    async def plan():
        return await manager.plan()

    @r.post("/plan/{item_id}/cancel")
    async def cancel_item(item_id: str):
        return {"item": await manager.cancel_item(item_id)}

    return r
