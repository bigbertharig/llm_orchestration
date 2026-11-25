"""Observed status: one row per GPU, reconciling leases with what the rig reports (pure functions)."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from .placement import FREE_MAX_USED_MB

VERDICTS = ("free", "claimed", "leased", "loading", "lease_idle", "stale", "unleased_busy")


def parse_time(value: Any) -> datetime | None:
    try:
        text = str(value).replace("Z", "+00:00")
        parsed = datetime.fromisoformat(text)
    except (TypeError, ValueError):
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def is_expired(lease: dict, now: datetime) -> bool:
    expires = parse_time(lease.get("expires_at"))
    return expires is None or expires <= now


def lease_port(lease: dict) -> int | None:
    port = (lease.get("metadata") or {}).get("port")
    if port is not None:
        return int(port)
    ports = lease.get("ports") or []
    return min(ports) if ports else None


def gpu_rows(obs: dict, leases: list[dict], claims: list[dict], now: datetime | None = None) -> list[dict]:
    now = now or datetime.now(timezone.utc)
    loading_leases = {p.get("lease") for p in obs.get("profiles", {}).values() if p.get("state") == "loading"}
    rows = []
    for gpu in obs.get("gpus", []):
        i = gpu["index"]
        lease = next((l for l in leases if i in l.get("gpu_ids", [])), None)
        claim = next((c for c in claims if i in c["gpu_ids"]), None)
        containers = [c for c in obs.get("containers", []) if i in c.get("gpus", [])]
        port = lease_port(lease) if lease else (containers[0].get("port") if containers else None)
        served = obs.get("ports", {}).get(str(port)) if port is not None else None
        if lease:
            expired = is_expired(lease, now)
            if containers:
                verdict = "stale" if expired else ("leased" if served else "loading")
            elif lease.get("lease_id") in loading_leases:
                verdict = "loading"
            else:
                verdict = "stale" if expired else "lease_idle"
        else:
            busy = containers or gpu.get("used_mb", 0) > FREE_MAX_USED_MB
            verdict = "unleased_busy" if busy else ("claimed" if claim else "free")
        rows.append({
            "gpu": i, "name": gpu.get("name"), "vram_mb": gpu.get("total_mb"),
            "verdict": verdict,
            "lease": None if not lease else {
                k: lease.get(k) for k in ("lease_id", "controller", "lane", "owner", "run_id", "expires_at")
            } | {"model": (lease.get("metadata") or {}).get("model"), "port": lease_port(lease)},
            "claim": claim["name"] if claim else None,
            "claim_info": None if not claim else {k: claim.get(k) for k in ("owner", "expires_at", "purpose", "model")
                                                  if claim.get(k)},
            "observed": {"used_mb": gpu.get("used_mb"), "util_pct": gpu.get("util_pct"),
                         "power_w": gpu.get("power_w"), "temp_c": gpu.get("temp_c"), "fan_pct": gpu.get("fan_pct"),
                         "power_limit_w": gpu.get("power_limit_w"), "containers": [c["name"] for c in containers],
                         "port": port, "serving": served},
        })
    return rows
