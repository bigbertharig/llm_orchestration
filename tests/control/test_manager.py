"""Rig manager: placement and status rules, plus the manager end to end against a fake `rig`."""

from __future__ import annotations

import json
import sys
import textwrap
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

import gguf
from control_fixture import create_app
from rig_control.manager import core
from rig_control.manager.placement import availability, place, port_conflict, port_for
from rig_control.manager.status import gpu_rows
from control_fixture import Settings

GPUS = [{"index": 0, "name": "3090", "total_mb": 24564, "used_mb": 15},
        {"index": 1, "name": "1060", "total_mb": 6144, "used_mb": 4},
        {"index": 2, "name": "1060", "total_mb": 6144, "used_mb": 4}]
SMALL = {"min_vram_mb": 5800, "max_vram_mb": 8192}
BIG = {"min_vram_mb": 20000}


def obs(**extra):
    return {"gpus": [dict(g) for g in GPUS], "containers": [], "ports": {}, "profiles": {}, "jobs": [], **extra}


# ------------------------------------------------------------------ placement and status (pure)

def test_smallest_fit_and_max_vram_keep_small_jobs_off_the_3090():
    free, _ = availability(obs(), [], [])
    assert place([SMALL], free) == [1]
    assert place([BIG], free) == [0]
    assert place([SMALL, SMALL, SMALL], free) is None  # only two 1060s; the 3090 is above max
    assert port_for([2, 1]) == 11435


def test_unleased_runtime_and_used_vram_make_a_card_unavailable():
    o = obs(containers=[{"name": "llama-lane-gpu1", "gpus": [1], "port": 11435}])
    o["gpus"][2]["used_mb"] = 3000
    free, why = availability(o, [], [])
    assert [g["index"] for g in free] == [0]
    assert "container llama-lane-gpu1" in why[1] and "3000 MB" in why[2]


def test_claims_exclude_cards_unless_named():
    claims = [{"name": "roo", "gpu_ids": [1]}]
    assert [g["index"] for g in availability(obs(), [], claims)[0]] == [0, 2]
    assert [g["index"] for g in availability(obs(), [], claims, claim="roo")[0]] == [1]
    with pytest.raises(KeyError):
        availability(obs(), [], claims, claim="nope")


def test_port_conflicts():
    assert "serving" in port_conflict(11435, obs(ports={"11435": "x.gguf"}), [])
    assert "reserved" in port_conflict(11435, obs(), [{"lease_id": "l", "ports": [11435]}])
    assert port_conflict(11434, obs(), []) is None


def test_status_verdicts():
    now = datetime.now(timezone.utc)
    future = (now + timedelta(minutes=5)).isoformat()
    past = (now - timedelta(minutes=5)).isoformat()
    o = obs(containers=[{"name": "a", "gpus": [0], "port": 11434}, {"name": "b", "gpus": [2], "port": 11436}],
            ports={"11434": "m.gguf"})
    leases = [{"lease_id": "l0", "gpu_ids": [0], "ports": [11434], "expires_at": future, "metadata": {"port": 11434}},
              {"lease_id": "l1", "gpu_ids": [1], "ports": [11435], "expires_at": past, "metadata": {"port": 11435}}]
    rows = {r["gpu"]: r["verdict"] for r in gpu_rows(o, leases, [], now)}
    assert rows == {0: "leased", 1: "stale", 2: "unleased_busy"}


# ------------------------------------------------------------------ manager end to end (fake rig)

FAKE_RIG = textwrap.dedent("""\
    import json, os, sys, time
    path = os.environ["FAKE_RIG_STATE"]
    if os.path.exists(path + ".down"):  # FAKE: the rig is unreachable (ssh exits 255)
        print("ssh: connect to host rig port 22: No route to host", file=sys.stderr); sys.exit(255)
    st = json.load(open(path))
    st.setdefault("controller", None); st.setdefault("plan_items", []); st.setdefault("runs", [])
    a = sys.argv[1:]
    def save():  # atomic: concurrent fake-rig calls never read a half-written state
        tmp = f"{path}.{os.getpid()}.tmp"
        json.dump(st, open(tmp, "w")); os.replace(tmp, path)
    def now_iso(): return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    def refresh():
        changed = False
        for p in st["profiles"].values():
            if p["state"] == "loading" and time.time() >= p["ready_at"]:
                p["state"] = "ready"; changed = True
        for i in st["plan_items"]:  # the executor: a running load is done once its profile serves
            prof = st["profiles"].get(i["args"].get("profile"))
            if i["state"] == "running" and i["kind"] == "load" and prof and prof["state"] in ("ready", "gone"):
                i["state"] = "done"; i["result"] = "ready"; changed = True
        return changed
    if refresh(): save()  # a read that changed nothing writes nothing (no lost updates between calls)
    def fail(kind, msg, **extra):
        print(json.dumps({"error": msg, "kind": kind, **extra})); sys.exit(1)
    def opts(xs):
        d, i = {}, 0
        while i < len(xs):
            if xs[i] == "--force": d["--force"] = True; i += 1
            else: d[xs[i]] = xs[i + 1] if i + 1 < len(xs) else None; i += 2
        return d
    def require(o):
        c = st["controller"]
        if c and o.get("--controller") != c["id"]:
            fail("not_controller", f"controller {o.get('--controller')!r} is not active (active: {c['id']})")
    if a[0] == "model-fit":
        from rig_control.manager import fit, models, placement
        q = json.loads(a[2])
        if q.get("placement_policy") not in (None, "fastest_fit", "smallest_fit"):
            fail("invalid", "invalid placement policy")
        try:
            r = models.resolve(q["model"], q.get("use", "benchmark"), q.get("profile"))
        except KeyError as e:
            fail("not_found", str(e))
        shape = fit.shape_for(r["gguf_path"], r["model"])
        full = {"gpus": st["gpus"], "containers": st["containers"]}
        if q.get("assume_idle"):
            free, why = [{**g, "used_mb": 0} for g in st["gpus"]], {}
        else:
            try:
                free, why = placement.availability(full, st["leases"], st["claims"], q.get("claim"))
            except KeyError as e:
                fail("not_found", str(e))
        bounds = r.get("context_bounds") or {}
        minimum = max(int(bounds.get("min") or fit.CTX_MIN_DEFAULT), int(q.get("ctx_min") or fit.CTX_MIN_DEFAULT))
        maximum = min(int(bounds.get("max") or shape.n_ctx_train), shape.n_ctx_train,
                      int((r.get("loading_stage") or {}).get("context_limit") or shape.n_ctx_train))
        mode = q.get("context_mode", "working")
        ctx = q.get("ctx")
        if ctx is not None and mode != "working":
            fail("invalid", "ctx is exact; context_mode applies only when ctx is omitted")
        if ctx is not None and (ctx < minimum or ctx > maximum):
            fail("invalid", f"ctx {ctx} exceeds maximum {maximum} or minimum {minimum}")
        if ctx is None and mode == "min": ctx = minimum
        if ctx is None and mode == "working": ctx = (r.get("runtime") or {}).get("ctx_size")
        kw = {"ctx": ctx, "ctx_min": minimum, "ctx_max": maximum,
              "ubatch": int(q.get("batch_size") or (r.get("runtime") or {}).get("batch_size") or 64),
              "parallel": int((r.get("runtime") or {}).get("parallel") or 1),
              "margin_mb": q.get("margin_mb", fit.MARGIN_MB_DEFAULT),
              "runtime_image": r.get("image")}
        ids = q.get("gpu_ids")
        count = q.get("cards")
        excluded = q.get("exclude_gpu_ids") or []
        free = [g for g in free if g["index"] not in excluded]
        if ids:
            blocked = {g: why.get(g, "unknown GPU") for g in ids if g not in {f["index"] for f in free}}
            if blocked: fail("conflict", "GPU(s) not available: " + str(blocked))
            result = fit.plan(shape, [g for g in free if g["index"] in ids], **kw)
        elif count:
            sets = fit.candidate_sets(free, int(count))
            if not sets: fail("capacity", "no capacity for requested cards")
            result = next((p for group in sets if (p := fit.plan(shape, group, **kw))["fits"]),
                          fit.plan(shape, sets[0], **kw))
        else:
            fastest = q.get("placement_policy", "fastest_fit" if q.get("use") == "chat" else "smallest_fit") == "fastest_fit"
            result = (fit.choose_chat(shape, free, **kw) if mode == "max" and fastest else
                      fit.plan(shape, free, **kw) if mode == "max" else
                      fit.choose_min(shape, free, **kw) if mode == "min" else
                      fit.choose_chat(shape, free, **kw) if fastest else
                      fit.choose_smallest(shape, free, **kw))
        if (not result or not result.get("fits")) and q.get("use") == "chat" and not ids and not count and ctx:
            host = (st.get("guard") or {}).get("mem_available_mb")
            if host is not None:
                for g in sorted(free, key=lambda g: -g["total_mb"]):
                    ram = fit.plan_ram(shape, g, ctx=ctx, ubatch=kw["ubatch"],
                                       mem_available_mb=host, margin_mb=kw["margin_mb"])
                    if ram["fits"]: result = ram; break
        if not result or not result.get("fits"):
            idle = [{**g, "used_mb": 0} for g in st["gpus"]]
            would_fit = fit.choose_smallest(shape, idle, **kw)
            if would_fit and not ids and not count and not q.get("assume_idle"):
                fail("capacity", "no capacity while GPUs are occupied or reserved")
            suggestions = []
            if ids and ctx and ctx > minimum:
                smaller = fit.plan(shape, [g for g in free if g["index"] in ids],
                                   **{**kw, "ctx": minimum})
                if smaller["fits"]:
                    suggestions.append(f"same GPU {ids} at minimum ctx {minimum} (context_mode=min)")
            fail("unfit", "model won't fit; fits on GPU 0 at another placement: " +
                 str((result or {}).get("reason")), suggestions=suggestions)
        if not ids and not count:
            result["chosen_by"] = ("policy: fastest available fit" if fastest else
                                   "policy: smallest available fit")
        result.update(registry_profile=r["profile"], status=r["status"],
                      use=q.get("use", "benchmark"), gguf_path=r["gguf_path"],
                      placement_policy=q.get("placement_policy") or ("fastest_fit" if q.get("use") == "chat" else "smallest_fit"),
                      ubatch=kw["ubatch"], context_mode=mode,
                      context_bounds={"min": minimum, "working": (r.get("runtime") or {}).get("ctx_size"),
                                      "max": maximum, "max_basis": bounds.get("max_basis", "gguf_train")})
        print(json.dumps({"fit": result, "source_sha256": os.getenv("FAKE_RIG_SOURCE_OVERRIDE") or
                          models.catalog()["source_sha256"]})); sys.exit(0)
    if a[0] == "fit-observation":
        row = json.loads(a[2])
        st.setdefault("fit_observations", []).append(row)
        save(); print(json.dumps({"recorded": True, "kind": row["kind"]})); sys.exit(0)
    def plan_report():
        return {"controller": st["controller"], "items": st["plan_items"], "version": st.get("plan_version", 1),
                "hold": st.get("plan_hold"), "boot_id": st.get("boot_id", "boot-1"),
                "next": [i for i in st["plan_items"] if i["state"] in ("queued", "running")], "runs": st["runs"]}
    if a[:2] == ["plan", "resume"]:
        o = opts(a[2:]); require(o)
        st["plan_hold"] = None; st["plan_version"] = st.get("plan_version", 1) + 1
        save(); print(json.dumps({"plan": plan_report()})); sys.exit(0)
    if a[:2] == ["plan", "restore"]:
        o = opts(a[2:]); doc = json.loads(o["--doc"])
        if doc.get("version", 0) <= st.get("plan_version", 1):
            fail("state", "the rig's plan is current")
        st["plan_items"] = doc["items"]; st["plan_version"] = doc["version"] + 1; st["restored"] = True
        save(); print(json.dumps({"plan": plan_report()})); sys.exit(0)
    if a[:2] == ["controller", "claim"]:
        o = opts(a[2:]); c = st["controller"]
        if c and c["id"] != o["--id"] and not o.get("--force") and not st.get("controller_stale"):
            fail("not_controller", f"controller {c['id']} (10.0.0.2) is active; use a handoff")
        prev = c["id"] if c and c["id"] != o["--id"] else None
        st["controller"] = {"id": o["--id"], "host": "test", "since": now_iso(), "heartbeat_at": now_iso(),
                            **({"took_over_from": prev} if prev else {})}
        st["controller_stale"] = False
        save(); print(json.dumps({"controller": st["controller"]})); sys.exit(0)
    if a[:2] in (["controller", "heartbeat"], ["controller", "release"]):
        o = opts(a[2:]); c = st["controller"]
        if not c or c["id"] != o["--id"]:
            fail("not_controller", f"controller {o['--id']!r} is not active (active: {c['id'] if c else 'none'})")
        if a[1] == "release": st["controller"] = None
        else: c["heartbeat_at"] = now_iso()
        save(); print(json.dumps({"controller": st["controller"]})); sys.exit(0)
    if a[:2] == ["plan", "add"]:
        o = opts(a[2:]); require(o)
        kind, args = o["--kind"], json.loads(o["--args"])
        item = {"id": f"{kind}-{len(st['plan_items'])}", "kind": kind, "args": args, "state": "running",
                "added_by": o["--controller"], "result": None}
        mode = os.environ.get("FAKE_DETACH", "ok")
        if kind == "job":  # the executor starts the run at once
            old = [r for r in st["runs"] if r["run"] == args["run"]]
            st["runs"] = [r for r in st["runs"] if r["run"] != args["run"]]
            st["runs"].append({**args, "unit": "rig-run-" + args["run"], "state": "running", "result": "",
                               "started_at": now_iso(), "ended_at": None,
                               "attempts": (old[0].get("attempts", 1) + 1) if old else 1})
            item.update(state="done", result="started")
            st["plan_items"].append(item); save(); print(json.dumps({"item": item})); sys.exit(0)
        pid = args["profile"]
        if kind == "load":
            st.setdefault("starts", []).append({"profile": pid, "gpus": ",".join(str(g) for g in args["gpus"]),
                                                "port": str(args["port"]), "lease": args["lease"],
                                                "cache_ram": str(args["cache_ram"])})
            if mode == "fail":
                item.update(state="failed", result="error: boom: cudaMalloc failed: out of memory")
            elif mode == "refuse":
                item.update(state="failed", result="error: manager placed it on a busy GPU")
            else:
                st["profiles"][pid] = {"port": int(args["port"]), "gpus": args["gpus"], "state": "loading",
                                       "lease": args["lease"], "ready_at": time.time() + 0.5,
                                       "model": args["spec"]["model"], "ctx_size": args["spec"]["ctx_size"]}
        else:
            st["profiles"].pop(pid, None); item.update(state="done", result="unloaded")
        st["plan_items"].append(item)
        save(); print(json.dumps({"item": item})); sys.exit(0)
    if a[:2] == ["plan", "cancel"]:
        o = opts(a[3:]); require(o)
        match = [i for i in st["plan_items"] if i["id"] == a[2]]
        if not match: fail("not_found", f"no plan item {a[2]}")
        match[0]["state"] = "cancelled"; save(); print(json.dumps({"item": match[0]})); sys.exit(0)
    if a[:2] in (["lease", "acquire"], ["lease", "preview"]):
        spec = json.loads(a[3])
        if a[1] == "acquire": require(opts(a[4:]))
        if "gpus_needed" in spec:
            pool = sorted(st["gpus"], key=lambda g: (g["total_mb"], g["index"]))
            occupied = {g for l in st["leases"] for g in l["gpu_ids"]}
            occupied |= {g for c in st.get("containers", []) for g in c["gpus"]}
            wanted_claim = (spec.get("metadata") or {}).get("claim")
            if wanted_claim and not any(c["name"] == wanted_claim for c in st["claims"]):
                fail("not_found", f"no claim {wanted_claim}")
            pool = [g for g in pool if g["index"] not in occupied and g.get("used_mb", 0) <= 1000
                    and (not wanted_claim or any(c["name"] == wanted_claim and g["index"] in c["gpu_ids"]
                                             for c in st["claims"]))
                    and not any(g["index"] in c["gpu_ids"] and c["name"] != wanted_claim for c in st["claims"])]
            chosen = {}
            for pos, req in sorted(enumerate(spec["gpus_needed"]), key=lambda item: -item[1]["min_vram_mb"]):
                match = next((g for g in pool if req["min_vram_mb"] <= g["total_mb"]
                              <= req.get("max_vram_mb", 1 << 30)), None)
                if not match: fail("capacity", f"no capacity for {spec['gpus_needed']}")
                chosen[pos] = match["index"]; pool.remove(match)
            gpus = [chosen[pos] for pos in range(len(chosen))]
        else:
            gpus = spec["gpu_ids"]
            running = {g for c in st.get("containers", []) for g in c["gpus"]}
            blocked = [g for g in gpus if g not in {row["index"] for row in st["gpus"]}
                       or g in running or next((row["used_mb"] for row in st["gpus"]
                                                if row["index"] == g), 1001) > 1000]
            if blocked: fail("conflict", f"GPU(s) not available: {blocked}")
        port = 11434 + min(gpus)
        if str(port) in st.get("ports", {}): fail("conflict", f"port {port} is already serving a runtime")
        spec["ports"] = [11434 + g for g in sorted(gpus)]
        spec["metadata"] = {**(spec.get("metadata") or {}), "port": port}
        for l in st["leases"]:
            if set(gpus) & set(l["gpu_ids"]):
                fail("conflict", f"GPU(s) {gpus} held by lease {l['lease_id']} ({l['controller']}/{l['owner']}, active)")
        for c in st["claims"]:
            if set(gpus) & set(c["gpu_ids"]) and (spec.get("metadata") or {}).get("claim") != c["name"]:
                fail("conflict", f"GPU {gpus} claimed by {c['name']}")
        if a[1] == "preview":
            print(json.dumps({"admitted": True, "gpu_ids": gpus, "ports": spec["ports"],
                              "port": port})); sys.exit(0)
        now = time.time()
        lease = {"lease_id": f"{spec['controller']}-{len(st['leases'])}-{int(now*1000)}", "controller": spec["controller"],
                 "owner": spec["owner"], "run_id": spec["run_id"], "gpu_ids": gpus, "ports": spec["ports"],
                 "metadata": spec.get("metadata") or {},
                 "acquired_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now)),
                 "expires_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now + spec["ttl_s"]))}
        st["leases"].append(lease); save(); print(json.dumps({"lease": lease})); sys.exit(0)
    if a[:2] in (["lease", "renew"], ["lease", "release"]):
        o = opts(a[3:]); require(o); match = [l for l in st["leases"] if l["lease_id"] == a[2]]
        if not match:
            fail("not_found", f"lease not found: {a[2]}")
        if o.get("--owner") != match[0]["owner"] and not o.get("--force"):
            fail("ownership", f"lease {a[2]} is owned by {match[0]['owner']}, not {o.get('--owner')}")
        if a[1] == "renew":
            match[0]["expires_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() + int(o["--ttl"])))
            save(); print(json.dumps({"lease": match[0]})); sys.exit(0)
        st["leases"].remove(match[0]); save(); print(json.dumps({"released": a[2]})); sys.exit(0)
    if a[:2] == ["claim", "add"]:
        o = opts(a[3:]); require(o); gpus = [int(g) for g in o["--gpus"].split(",")]
        for c in st["claims"]:
            if c["name"] == a[2] or set(gpus) & set(c["gpu_ids"]):
                fail("conflict", f"GPU {gpus} already claimed by {c['name']}")
        claim = {"name": a[2], "gpu_ids": gpus, "note": o.get("--note", ""), "owner": o.get("--owner", "")}
        if o.get("--ttl"):
            claim["ttl_s"] = int(o["--ttl"])
            claim["expires_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() + int(o["--ttl"])))
        for k in ("--purpose", "--model"):
            if o.get(k): claim[k[2:]] = o[k]
        st["claims"].append(claim); save(); print(json.dumps({"claim": claim})); sys.exit(0)
    if a[:2] in (["run", "stop"], ["run", "dismiss"]):
        o = opts(a[3:]); require(o)
        match = [r for r in st["runs"] if r["run"] == a[2]]
        if not match: fail("not_found", f"no run {a[2]}")
        match[0].update(state="stopped" if a[1] == "stop" else "dismissed", ended_at=now_iso(),
                        result=f"{a[1]} {o.get('--reason', '')}")
        save(); print(json.dumps({"run": match[0]})); sys.exit(0)
    if a[:2] == ["claim", "extend"]:
        o = opts(a[3:]); require(o); gpus = [int(g) for g in o["--gpus"].split(",")]
        match = [c for c in st["claims"] if c["name"] == a[2]]
        if not match: fail("not_found", f"no claim {a[2]} (expired or removed)")
        for c in st["claims"]:
            if c is not match[0] and set(gpus) & set(c["gpu_ids"]):
                fail("conflict", f"GPU {gpus} already claimed by {c['name']}")
        match[0]["gpu_ids"] = sorted(set(match[0]["gpu_ids"]) | set(gpus))
        st.setdefault("extends", []).append({"name": a[2], "gpus": gpus})
        save(); print(json.dumps({"claim": match[0]})); sys.exit(0)
    if a[:2] == ["claim", "renew"]:
        o = opts(a[3:]); require(o)
        match = [c for c in st["claims"] if c["name"] == a[2]]
        if not match: fail("not_found", f"no claim {a[2]} (expired or removed)")
        st["renewed"] = st.get("renewed", 0) + 1
        save(); print(json.dumps({"claim": match[0]})); sys.exit(0)
    if a[0] == "kill":
        o = opts(a[1:]); require(o); gpus = {int(g) for g in o["--gpus"].split(",")}
        stopped = [c["name"] for c in st["containers"] if gpus & set(c["gpus"])]
        st["containers"] = [c for c in st["containers"] if not gpus & set(c["gpus"])]
        for k, p in st["profiles"].items():
            if gpus & set(p["gpus"]): stopped.append("rig-remote-" + k); p["state"] = "gone"
        released = [l["lease_id"] for l in st["leases"] if gpus & set(l["gpu_ids"])]
        st["leases"] = [l for l in st["leases"] if not gpus & set(l["gpu_ids"])]
        dropped = [c["name"] for c in st["claims"] if gpus & set(c["gpu_ids"])]
        st["claims"] = [c for c in st["claims"] if not gpus & set(c["gpu_ids"])]
        save(); print(json.dumps({"gpus": sorted(gpus), "stopped": stopped, "released": released,
                                  "claims_dropped": dropped})); sys.exit(0)
    if a[:2] == ["claim", "remove"]:
        require(opts(a[3:]))
        match = [c for c in st["claims"] if c["name"] == a[2]]
        if not match:
            fail("not_found", f"no claim {a[2]}")
        st["claims"].remove(match[0]); save(); print(json.dumps({"released": match[0]})); sys.exit(0)
    if a == ["observe"]:
        ctrs = list(st["containers"]) + [{"name": "rig-remote-" + k, "gpus": p["gpus"], "port": p["port"]}
                                         for k, p in st["profiles"].items() if p["state"] != "gone"]
        ports = {str(p["port"]): k + ".gguf" for k, p in st["profiles"].items() if p["state"] == "ready"}
        profiles = st["profiles"]
        # FAKE_STALE_READS=N: the first N snapshots after a load is done still show its profile loading
        # (a rig observe that read the profiles before the plan).
        stale = int(os.environ.get("FAKE_STALE_READS", "0"))
        if st.get("stale_reads", 0) < stale and any(i["kind"] == "load" and i["state"] == "done"
                                                    for i in st["plan_items"]):
            st["stale_reads"] = st.get("stale_reads", 0) + 1; save()
            profiles = {k: {**v, "state": "loading"} if v["state"] == "ready" else v for k, v in profiles.items()}
        print(json.dumps({"observed_at": "now", "mode": "remote" if st["profiles"] else "idle",
                          "gpus": st["gpus"], "containers": ctrs, "ports": ports, "profiles": profiles,
                          "jobs": st["jobs"], "last_error": st.get("last_error"),
                          "leases": st["leases"], "claims": st["claims"], "plan": plan_report(),
                          "guard": st.get("guard", {"level": "ok", "mem_available_mb": 20000})})); sys.exit(0)
    if a[:2] == ["status", "--json"]:
        print(json.dumps({"mode": "remote", "profiles": [{"id": k, **p} for k, p in st["profiles"].items()]}))
        sys.exit(0)
    if a[0] == "stop":
        st["profiles"].pop(a[1], None); save(); print("unloaded"); sys.exit(0)
    sys.exit(64)
""")


def any_llama(request: httpx.Request) -> httpx.Response:
    return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}], "port": request.url.port})


@pytest.fixture
def menv(tmp_path, monkeypatch):
    (tmp_path / "rig.py").write_text(FAKE_RIG)
    state = tmp_path / "state.json"
    state.write_text(json.dumps({"gpus": GPUS, "containers": [], "profiles": {}, "jobs": [],
                                 "leases": [], "claims": []}))
    # Published profiles name no cards: the manager's fit places them from the GGUF sizes in its catalog
    # (small and worker fit one 1060; big needs the 3090).
    gguf.write_catalog(tmp_path / "model_catalog.json", {
        "s:7b": ("active", gguf.dense(tmp_path, n_layer=10, layer_mb=300, name="s.gguf")),
        "b:20b": ("active", gguf.dense(tmp_path, n_layer=35, layer_mb=400, ctx_train=131072, name="b.gguf")),
        "w:7b": ("active", gguf.dense(tmp_path, n_layer=10, layer_mb=300, name="w.gguf"))})
    monkeypatch.setenv("MODEL_CATALOG", str(tmp_path / "model_catalog.json"))
    monkeypatch.setenv("CHAT_DATA_ROOT", str(tmp_path / "chat"))
    profiles = tmp_path / "remote_profiles.json"
    profiles.write_text(json.dumps({"remote_profiles": [
        {"id": "small", "model": "s:7b", "registry_profile": "default", "ctx_size": 16384,
         "status": "active", "use": "chat", "good_for": ["chatbot"]},
        {"id": "big", "model": "b:20b", "registry_profile": "default", "ctx_size": 65536},
        {"id": "worker", "model": "w:7b", "registry_profile": "default", "ctx_size": 8192, "use": "worker"},
    ]}))
    monkeypatch.setenv("FAKE_RIG_STATE", str(state))
    monkeypatch.setenv("PYTHONPATH", str(Path(__file__).resolve().parents[2] / "src"))
    monkeypatch.setattr(core, "LOAD_POLL_S", 0.1)
    monkeypatch.setattr(core, "OBSERVE_CACHE_S", 0.0)
    settings = Settings(
        mirror_path=tmp_path / "copy" / "rig_report.json",
        rig_command=(sys.executable, str(tmp_path / "rig.py")), profiles_path=profiles,
        upstream_host="127.0.0.1", model_prefix="gpu-rig", bind="127.0.0.1", port=0,
        start_timeout_seconds=30, request_timeout_seconds=30, manager_enabled=True,
    )
    app = create_app(settings, upstream_transport=httpx.MockTransport(any_llama), run_reconciler=False)

    def rig_state():
        return json.loads(state.read_text())

    def write_state(st):
        state.write_text(json.dumps(st))

    def leases():
        return rig_state()["leases"]

    with TestClient(app) as c:
        yield {"client": c, "rig": rig_state, "write": write_state, "leases": leases, "app": app, "state": str(state),
               "copy": tmp_path / "copy" / "rig_report.json", "rig_py": tmp_path / "rig.py"}


def chat(c, model):
    return c.post("/v1/chat/completions", json={"model": model, "messages": [{"role": "user", "content": "hi"}]})


def test_cold_request_takes_a_lease_and_queues_one_load(menv):
    r = chat(menv["client"], "gpu-rig/small")
    assert r.status_code == 200 and r.json()["port"] == 11434
    (lease,) = menv["leases"]()
    assert lease["controller"] == "interactive" and lease["gpu_ids"] == [0]
    assert lease["metadata"]["port"] == 11434 and lease["ports"] == [11434]
    (start,) = menv["rig"]()["starts"]
    assert start == {"profile": "small", "gpus": "0", "port": "11434", "lease": lease["lease_id"], "cache_ram": "4096"}
    assert chat(menv["client"], "gpu-rig/small").status_code == 200  # warm: no second job
    assert len(menv["rig"]()["starts"]) == 1


def test_cards_busy_without_a_lease_are_skipped(menv):
    st = menv["rig"]()
    st["containers"] = [{"name": "llama-lane-gpu1", "gpus": [1], "port": 11435}]
    menv["write"](st)
    assert chat(menv["client"], "gpu-rig/small").json()["port"] == 11434
    rows = {g["gpu"]: g["verdict"] for g in menv["client"].get("/v1/manager/status").json()["gpus"]}
    assert rows[1] == "unleased_busy" and rows[0] == "leased"


def test_published_chat_load_uses_max_fit_on_fastest_card(menv):
    c = menv["client"]
    preview = c.post("/v1/manager/models/fit",
                     json={"model": "s:7b", "use": "chat", "profile": "default",
                           "ctx_min": 16384, "context_mode": "max"}).json()
    assert preview["fits"] and preview["gpus"] == [0]
    assert preview["ctx"] > 16384
    assert chat(c, "gpu-rig/small").status_code == 200
    load_item = next(i for i in menv["rig"]()["plan_items"] if i["kind"] == "load")
    assert load_item["args"]["spec"]["ctx_size"] == preview["ctx"]


def test_basic_chat_busy_signal_blocks_cold_load_even_with_free_1060s(menv):
    c = menv["client"]
    idle = c.get("/v1/manager/chat/availability").json()
    assert idle["busy"] is False
    st = menv["rig"]()
    st["containers"] = [{"name": "benchmark-gpu0", "gpus": [0], "port": 11434}]
    menv["write"](st)
    busy = c.get("/v1/manager/chat/availability").json()
    assert busy["busy"] is True and "active runtime" in busy["reason"]
    result = c.post('/test/prepare/small')
    assert result.status_code == 409 and 'rig busy' in result.json()['error']['message']
    assert menv['rig']().get('starts', []) == []


def test_basic_chat_can_continue_its_own_ready_profile(menv):
    c = menv['client']
    first = c.post('/test/prepare/small')
    assert first.status_code == 200, first.text
    availability = c.get('/v1/manager/chat/availability?profile_id=small').json()
    assert availability['busy'] is False and availability['requested_ready'] is True
    assert c.post('/test/prepare/small').status_code == 200
    assert len(menv['rig']()['starts']) == 1


def test_chat_uses_a_1060_when_the_3090_is_busy(menv):
    st = menv["rig"]()
    st["containers"] = [{"name": "llama-lane-gpu0", "gpus": [0], "port": 11434}]
    menv["write"](st)
    assert chat(menv["client"], "gpu-rig/small").json()["port"] == 11435
    (lease,) = menv["leases"]()
    assert lease["gpu_ids"] == [1]


def test_claimed_cards_leave_the_pool(menv):
    c = menv["client"]
    assert c.post("/v1/manager/claims", json={"name": "bench", "gpu_ids": [0, 1, 2]}).status_code == 201
    r = chat(c, "gpu-rig/small")  # every card claimed: fits once they free, so 503, not "won't fit"
    assert r.status_code == 503 and "no capacity" in r.json()["error"]["message"]
    assert c.post("/v1/manager/profiles/small/load", json={"claim": "bench"}).json()["port"] == 11434
    assert c.post("/v1/manager/claims", json={"name": "x", "gpu_ids": [2]}).status_code == 409
    assert c.post("/v1/manager/claims/bench/release").status_code == 200


def test_benchmark_claim_explains_idle_and_loaded_cards(menv):
    c = menv["client"]
    created = c.post("/v1/manager/claims", json={
        "name": "bench-session", "gpu_ids": [0, 1, 2],
        "owner": "benchmarking", "purpose": "Nemo fit and two chats",
    })
    assert created.status_code == 201, created.text
    from rig_control.manager.placement import availability
    rig = menv["rig"]()
    rig["gpus"][0]["used_mb"] = 18000
    _, why = availability(rig, menv["leases"](), [created.json()["claim"]])
    for card in (0, 1, 2):
        assert "bench-session" in why[card]
        assert "benchmarking" in why[card]
        assert "Nemo fit and two chats" in why[card]
    assert "busy without a lease" in why[0]
    assert "busy without a lease" not in why[1]
    rows = {row["gpu"]: row for row in c.get("/v1/manager/status").json()["gpus"]}
    assert rows[1]["verdict"] == "claimed"
    assert rows[1]["claim_info"]["purpose"] == "Nemo fit and two chats"
    wrong = c.post("/v1/manager/claims/bench-session/extend",
                   json={"gpu_ids": [0, 1, 2], "owner": "someone-else"})
    assert wrong.status_code == 403
    assert c.post("/v1/manager/claims/bench-session/release").status_code == 200
    first = c.post("/v1/manager/claims", json={
        "name": "expand", "gpu_ids": [0], "owner": "benchmarking", "purpose": "next stage",
    })
    assert first.status_code == 201
    extended = c.post("/v1/manager/claims/expand/extend",
                      json={"gpu_ids": [0, 1, 2], "owner": "benchmarking"})
    assert extended.status_code == 200 and extended.json()["claim"]["gpu_ids"] == [0, 1, 2]


def test_published_profile_load_can_pin_gpus(menv):
    c = menv["client"]
    r = c.post("/v1/manager/profiles/small/load", json={"gpu_ids": [1]})
    assert r.status_code == 200, r.text
    assert menv["leases"]()[0]["gpu_ids"] == [1]
    assert menv["rig"]()["starts"][0]["gpus"] == "1"
    assert c.post("/v1/manager/profiles/small/load", json={"gpu_ids": [0]}).status_code == 409
    assert c.post("/v1/manager/profiles/small/load", json={"gpu_ids": []}).status_code == 400


def test_failed_rig_job_releases_the_lease_and_reports_the_log(menv, monkeypatch):
    monkeypatch.setenv("FAKE_DETACH", "fail")
    r = chat(menv["client"], "gpu-rig/small")
    assert r.status_code == 502 and "out of memory" in r.json()["error"]["message"]
    assert menv["leases"]() == []


def test_done_load_seen_loading_once_is_not_failed(menv, monkeypatch):
    monkeypatch.setenv("FAKE_STALE_READS", "1")
    r = chat(menv["client"], "gpu-rig/small")
    assert r.status_code == 200, r.text
    assert len(menv["leases"]()) == 1


def test_done_load_that_never_serves_fails_and_releases(menv, monkeypatch):
    monkeypatch.setenv("FAKE_STALE_READS", "99")
    r = chat(menv["client"], "gpu-rig/small")
    assert r.status_code == 502 and "not serving" in r.json()["error"]["message"]
    assert menv["leases"]() == []


def test_rig_refusal_releases_the_lease(menv, monkeypatch):
    monkeypatch.setenv("FAKE_DETACH", "refuse")
    r = chat(menv["client"], "gpu-rig/small")
    assert r.status_code == 502 and "busy GPU" in r.json()["error"]["message"]
    assert menv["leases"]() == []


def test_explicit_gpus_conflict_names_the_holder(menv):
    c = menv["client"]
    body = {"controller": "benchmarks", "owner": "lanes", "run_id": "r1", "gpu_ids": [0], "model": "x"}
    first = c.post("/v1/manager/leases", json=body)
    assert first.status_code == 201 and first.json()["port"] == 11434
    second = c.post("/v1/manager/leases", json={**body, "run_id": "r2"})
    assert second.status_code == 409 and "benchmarks/lanes" in second.json()["error"]["message"]
    r = chat(c, "gpu-rig/big")
    assert r.status_code == 503
    lid = first.json()["lease"]["lease_id"]
    assert c.post(f"/v1/manager/leases/{lid}/release", json={"owner": "someone"}).status_code == 403
    assert c.post(f"/v1/manager/leases/{lid}/release", json={"owner": "lanes"}).status_code == 200


def test_lease_preview_uses_rig_admission_without_reserving_cards(menv):
    c = menv["client"]
    request = {"gpus_needed": [{"min_vram_mb": 5000, "max_vram_mb": 8192}]}
    response = c.post("/v1/manager/leases/preview", json=request)
    assert response.status_code == 200
    assert response.json() == {"admitted": True, "gpu_ids": [1], "ports": [11435], "port": 11435}
    assert menv["leases"]() == []
    refused = c.post("/v1/manager/leases/preview", json={
        "gpus_needed": [{"min_vram_mb": 30000}]})
    assert refused.status_code == 503
    assert "no capacity" in refused.json()["error"]["message"]
    assert menv["leases"]() == []


def test_model_fit_refuses_divergent_library_revisions(menv, monkeypatch):
    monkeypatch.setenv("FAKE_RIG_SOURCE_OVERRIDE", "different-revision")
    response = menv["client"].post("/v1/manager/models/fit", json={"model": "s:7b"})
    assert response.status_code == 503
    assert "libraries differ" in response.json()["error"]["message"]
    assert menv["leases"]() == []


def test_force_load_rejects_unknown_model_before_stopping_occupant(menv):
    c = menv["client"]
    assert c.post("/v1/manager/models/load", json={"model": "s:7b", "use": "chat",
                                                  "gpu_ids": [0]}).status_code == 200
    before = menv["rig"]()
    result = c.post("/v1/manager/models/load", json={"model": "missing:model", "use": "chat",
                                                     "gpu_ids": [0], "force": True})
    assert result.status_code == 404
    after = menv["rig"]()
    assert after["profiles"] == before["profiles"]
    assert after["leases"] == before["leases"]
    assert after["starts"] == before["starts"]


def test_all_cards_occupied_refuses_suggestion_without_a_rig_action(menv):
    c = menv["client"]
    st = menv["rig"]()
    for gpu in st["gpus"]:
        gpu["used_mb"] = 3000 if gpu["index"] else 23000
    menv["write"](st)
    before = menv["rig"]()

    preview = c.post("/v1/manager/models/fit", json={"model": "s:7b", "use": "chat"})
    request = c.post("/v1/manager/models/load", json={"model": "s:7b", "use": "chat"})
    chat_status = c.get("/v1/manager/chat/availability").json()

    assert preview.status_code == request.status_code == 503
    assert "no capacity" in preview.json()["error"]["message"]
    assert chat_status["busy"] and "GPU memory in use" in chat_status["reason"]
    after = menv["rig"]()
    assert after["profiles"] == before["profiles"] == {}
    assert after["leases"] == before["leases"] == []
    assert after.get("starts", []) == before.get("starts", []) == []
    assert after.get("plan_items", []) == before.get("plan_items", []) == []


def test_owner_stop_frees_only_its_card_then_new_request_rechecks(menv):
    c = menv["client"]
    first = c.post("/v1/manager/models/load", json={"model": "s:7b", "use": "chat",
                                                  "gpu_ids": [0], "owner": "stuck-worker"})
    other = c.post("/v1/manager/models/load", json={"model": "w:7b", "use": "worker",
                                                  "gpu_ids": [1], "owner": "other-worker"})
    assert first.status_code == other.status_code == 200
    first_id, other_id = first.json()["profile"], other.json()["profile"]
    blocked = c.post("/v1/manager/models/fit", json={"model": "s:7b", "use": "chat", "gpu_ids": [0]})
    assert blocked.status_code == 409

    stopped = c.post("/v1/manager/owners/stuck-worker/cancel", json={"reason": "test stuck worker"})
    assert stopped.status_code == 200
    assert first_id in stopped.json()["profiles_unloaded"]
    snapshot = menv["rig"]()
    assert first_id not in snapshot["profiles"]
    assert other_id in snapshot["profiles"]
    assert {l["metadata"].get("requested_by") for l in snapshot["leases"]} == {"other-worker"}

    offer = c.post("/v1/manager/models/fit", json={"model": "s:7b", "use": "chat", "gpu_ids": [0]})
    assert offer.status_code == 200 and offer.json()["gpus"] == [0]
    replacement = c.post("/v1/manager/models/load", json={"model": "s:7b", "use": "chat",
                                                        "gpu_ids": [0], "owner": "replacement"})
    assert replacement.status_code == 200
    assert other_id in menv["rig"]()["profiles"]


def test_expired_foreign_lease_stays_occupied_until_owner_releases_it(menv):
    c = menv["client"]
    created = c.post("/v1/manager/leases", json={"controller": "benchmarks", "owner": "speed-test",
                                                  "run_id": "speed-1", "gpu_ids": [2]})
    assert created.status_code == 201
    lease_id = created.json()["lease"]["lease_id"]
    st = menv["rig"]()
    st["leases"][0]["expires_at"] = "2000-01-01T00:00:00Z"
    menv["write"](st)

    blocked = c.post("/v1/manager/models/fit", json={"model": "s:7b", "use": "chat", "gpu_ids": [2]})
    assert blocked.status_code == 409
    assert lease_id in blocked.json()["error"]["message"]
    assert c.post("/v1/manager/reconcile").status_code == 200
    assert [l["lease_id"] for l in menv["leases"]()] == [lease_id]
    assert c.post(f"/v1/manager/leases/{lease_id}/release", json={"owner": "speed-test"}).status_code == 200
    free = c.post("/v1/manager/models/fit", json={"model": "s:7b", "use": "chat", "gpu_ids": [2]})
    assert free.status_code == 200 and free.json()["gpus"] == [2]


def test_release_refuses_while_a_runtime_runs(menv):
    c = menv["client"]
    lid = c.post("/v1/manager/leases", json={"controller": "benchmarks", "owner": "lanes", "run_id": "r",
                                             "gpu_ids": [2]}).json()["lease"]["lease_id"]
    st = menv["rig"]()
    st["containers"] = [{"name": "llama-lane-gpu2", "gpus": [2], "port": 11436}]
    menv["write"](st)
    r = c.post(f"/v1/manager/leases/{lid}/release", json={"owner": "lanes"})
    assert r.status_code == 409 and "llama-lane-gpu2" in r.json()["error"]["message"]


def test_unload_releases_the_lease(menv):
    c = menv["client"]
    assert chat(c, "gpu-rig/small").status_code == 200
    out = c.post("/v1/manager/profiles/small/unload").json()
    assert out["unloaded"] == "small" and out["released"]
    assert menv["leases"]() == []


def test_reconcile_renews_backed_and_releases_orphaned_leases(menv, monkeypatch):
    c = menv["client"]
    assert chat(c, "gpu-rig/small").status_code == 200
    orphan = c.post("/v1/manager/leases", json={"controller": "interactive", "owner": core.OWNER,
                                                "run_id": "profile-gone", "gpus_needed": [SMALL]}).json()["lease"]
    monkeypatch.setattr(core, "ORPHAN_GRACE_S", 0)
    out = c.post("/v1/manager/reconcile").json()
    assert out["released"] == [orphan["lease_id"]] and len(out["renewed"]) == 1
    assert [l["run_id"] for l in menv["leases"]()] == ["model-small"]


def test_cache_follows_the_use_and_the_rig_free_ram(menv):
    from rig_control.manager.memory import cache_ram_mib
    assert cache_ram_mib("chat", 20000) == 4096
    assert cache_ram_mib("chat", 10000) == 1792      # bounded by what is left above the reserve
    assert cache_ram_mib("chat", 6000) == 0
    assert cache_ram_mib("chat", None) == 0
    assert cache_ram_mib("worker", 30000) == 0 and cache_ram_mib("benchmark", 30000) == 0
    c = menv["client"]
    assert chat(c, "gpu-rig/worker").status_code == 200
    st = menv["rig"]()
    st["guard"] = {"level": "ok", "mem_available_mb": 9000}
    menv["write"](st)
    assert chat(c, "gpu-rig/small").status_code == 200
    starts = {s["profile"]: s["cache_ram"] for s in menv["rig"]()["starts"]}
    assert starts == {"worker": "0", "small": "768"}


def test_gone_runtime_is_cleared_and_its_lease_released(menv, monkeypatch):
    c = menv["client"]
    assert chat(c, "gpu-rig/small").status_code == 200
    st = menv["rig"]()
    st["profiles"]["small"]["state"] = "gone"   # the rig guard stopped it
    menv["write"](st)
    monkeypatch.setattr(core, "ORPHAN_GRACE_S", 0)
    out = c.post("/v1/manager/reconcile").json()
    assert len(out["released"]) == 1
    assert menv["rig"]()["profiles"] == {} and menv["leases"]() == []


# ------------------------------------------------------------------ one active controller

def test_cold_request_claims_control_and_queues_the_load_in_the_rig_plan(menv):
    assert chat(menv["client"], "gpu-rig/small").status_code == 200
    st = menv["rig"]()
    assert st["controller"]["id"] == "controller"
    (item,) = st["plan_items"]
    assert item["kind"] == "load" and item["state"] == "done" and item["added_by"] == "controller"
    status = menv["client"].get("/v1/manager/status").json()
    assert status["controller"]["in_control"] and status["plan"]["controller"]["id"] == "controller"


def test_read_only_while_another_controller_is_active_then_handoff(menv):
    st = menv["rig"]()
    st["controller"] = {"id": "pi", "host": "10.0.0.2", "since": "t", "heartbeat_at": "t"}
    menv["write"](st)
    c = menv["client"]
    r = chat(c, "gpu-rig/small")
    assert r.status_code == 409 and "read-only" in r.json()["error"]["message"]
    assert menv["leases"]() == []
    ctl = c.get("/v1/manager/controller").json()
    assert ctl["in_control"] is False and ctl["active"]["id"] == "pi"
    taken = c.post("/v1/manager/controller/claim", json={"force": True}).json()
    assert taken["taken"] and taken["active"]["took_over_from"] == "pi"
    assert chat(c, "gpu-rig/small").status_code == 200


def test_stale_controller_is_taken_over(menv):
    st = menv["rig"]()
    st["controller"] = {"id": "pi", "host": "10.0.0.2", "since": "t", "heartbeat_at": "old"}
    st["controller_stale"] = True
    menv["write"](st)
    assert chat(menv["client"], "gpu-rig/small").status_code == 200
    assert menv["rig"]()["controller"]["took_over_from"] == "pi"


def test_losing_control_makes_the_manager_read_only(menv):
    c = menv["client"]
    assert chat(c, "gpu-rig/small").status_code == 200
    st = menv["rig"]()
    st["controller"] = {"id": "desktop", "host": "10.0.0.4", "since": "t", "heartbeat_at": "t"}
    menv["write"](st)
    out = c.post("/v1/manager/reconcile").json()
    assert out["in_control"] is False and "lost control" in out["note"]
    assert c.post("/v1/manager/claims", json={"name": "x", "gpu_ids": [2]}).status_code == 409


def test_unload_is_a_plan_item(menv):
    c = menv["client"]
    assert chat(c, "gpu-rig/small").status_code == 200
    out = c.post("/v1/manager/profiles/small/unload").json()
    assert out["state"] == "done" and out["released"]
    kinds = [i["kind"] for i in menv["rig"]()["plan_items"]]
    assert kinds == ["load", "unload"]


def test_reconcile_leaves_a_lease_alone_while_its_unload_runs(menv, monkeypatch):
    """2026-10-18: a lane's unload got 404 because the reconciler released the lease mid-unload."""
    c = menv["client"]
    manager = menv["app"].state.manager
    assert chat(c, "gpu-rig/small").status_code == 200
    (lease,) = menv["leases"]()
    monkeypatch.setattr(core, "ORPHAN_GRACE_S", 0)
    real_wait = core.Manager._wait_item
    seen = {}

    async def reconcile_mid_unload(self, *a, **kw):
        done = await real_wait(self, *a, **kw)  # the rig has unloaded the profile: the lease looks unbacked
        seen["reconcile"] = await self.reconcile_once()
        return done

    monkeypatch.setattr(core.Manager, "_wait_item", reconcile_mid_unload)
    r = c.post("/v1/manager/profiles/small/unload")
    assert r.status_code == 200 and r.json()["released"] == lease["lease_id"]
    assert lease["lease_id"] not in seen["reconcile"]["released"]  # the unload released it, not the reconciler
    assert menv["leases"]() == [] and manager._in_flight == set()


# ------------------------------------------------------------------ shared report (rule 1c)

def test_the_controller_keeps_a_copy_and_answers_from_it_when_the_rig_is_gone(menv):
    c = menv["client"]
    assert c.get("/v1/manager/status").status_code == 200
    copy = json.loads(menv["copy"].read_text())
    assert copy["plan"]["version"] == 1 and copy["gpus"]
    menv["rig_py"].write_text("import sys; print('ssh: connect to host 10.0.0.3: No route to host'); sys.exit(255)")
    r = c.get("/v1/manager/status")
    assert r.status_code == 200 and r.json()["from_copy"] and not r.json()["rig"]["reachable"]
    assert "rig unreachable" in r.json()["rig"]["events"][-1]["event"]


def test_a_reboot_hold_is_reviewed_and_resumed(menv, monkeypatch):
    monkeypatch.setattr(core, "ORPHAN_GRACE_S", 0)
    c = menv["client"]
    manager = menv["app"].state.manager
    assert chat(c, "gpu-rig/small").status_code == 200
    st = menv["rig"]()
    st.update(boot_id="boot-2", plan_hold={"reason": "boot", "since": "now", "interrupted": ["load-0"]},
              profiles={})
    st["plan_items"][0].update(state="interrupted", result="interrupted: the rig restarted while it ran (load small)")
    menv["write"](st)
    import asyncio
    out = asyncio.run(manager.reconcile_once())
    assert out["in_control"] and menv["rig"]()["plan_hold"] is None
    events = [e["event"] for e in manager.events]
    assert any("rig rebooted" in e for e in events)
    assert any("resumed; interrupted, not retried: load-0" in e for e in events)
    assert menv["leases"]() == []  # the interrupted load left an unbacked lease: released


def test_a_reboot_does_not_replay_queued_model_lifecycle_requests(menv):
    manager = menv["app"].state.manager
    st = menv["rig"]()
    st.update(boot_id="boot-2", plan_hold={"reason": "boot", "since": "now", "interrupted": []})
    st.setdefault("plan_items", []).extend([
        {"id": "load-stale", "kind": "load", "state": "queued", "args": {"profile": "small"}},
        {"id": "unload-stale", "kind": "unload", "state": "queued", "args": {"profile": "small"}},
        {"id": "job-durable", "kind": "job", "state": "queued", "args": {"run": "explicit-job"}},
    ])
    menv["write"](st)

    import asyncio
    asyncio.run(manager.reconcile_once())

    items = {item["id"]: item for item in menv["rig"]()["plan_items"]}
    assert items["load-stale"]["state"] == "cancelled"
    assert items["unload-stale"]["state"] == "cancelled"
    assert items["job-durable"]["state"] == "queued"
    assert menv["rig"]()["plan_hold"] is None
    assert any("stale model requests cancelled: load-stale, unload-stale" in event["event"]
               for event in manager.events)


def test_a_rig_that_lost_its_plan_gets_the_controller_copy_back(menv):
    c = menv["client"]
    manager = menv["app"].state.manager
    assert chat(c, "gpu-rig/small").status_code == 200
    st = menv["rig"]()
    saved = len(st["plan_items"])
    manager.mirror["plan"]["version"] = 40
    st.update(plan_items=[], plan_version=2, plan_hold={"reason": "boot", "since": "now", "interrupted": []})
    menv["write"](st)
    import asyncio
    asyncio.run(manager.reconcile_once())
    st = menv["rig"]()
    assert st.get("restored") and len(st["plan_items"]) == saved and st["plan_hold"] is None
