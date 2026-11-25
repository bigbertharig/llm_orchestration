"""Placement rules (pure functions over a `rig observe` snapshot, the leases and the claims).

- A card is available only if no lease holds it, nothing runs on it (no llama container, VRAM
  nearly empty), and it is unclaimed, or claimed by the claim the request names.
- `gpus_needed` entries are {"min_vram_mb": N} with optional {"max_vram_mb": M}; each takes the
  smallest available card that fits, so big cards stay free for requests that need them.
- The port is 11434 + the lowest GPU index in the placement (the rig's existing convention). A
  lease reserves 11434 + every GPU it holds, so two leases can never get the same port.
"""

from __future__ import annotations

from typing import Any

PORT_BASE = 11434
FREE_MAX_USED_MB = 1000


def port_for(gpu_ids: list[int]) -> int:
    return PORT_BASE + min(gpu_ids)


def lease_ports(gpu_ids: list[int]) -> list[int]:
    return [PORT_BASE + g for g in sorted(gpu_ids)]


def availability(obs: dict, leases: list[dict], claims: list[dict],
                 claim: str | None = None) -> tuple[list[dict], dict[int, str]]:
    """(available cards, reason each other card is unavailable)."""
    held = {g: lease for lease in leases for g in lease.get("gpu_ids", [])}
    claimed = {g: c for c in claims for g in c["gpu_ids"]}
    running = {g: c for c in obs.get("containers", []) for g in c.get("gpus", [])}
    if claim is not None and not any(c["name"] == claim for c in claims):
        raise KeyError(f"no claim {claim}")
    free, why = [], {}
    def claim_label(item):
        owner = item.get("owner") or item.get("by") or "unknown owner"
        purpose = item.get("purpose") or item.get("note") or "unspecified purpose"
        return f"claimed by {item['name']} (owner {owner}; purpose {purpose})"

    for gpu in obs.get("gpus", []):
        i = gpu["index"]
        if i in held:
            lease = held[i]
            why[i] = f"leased by {lease.get('controller')}/{lease.get('owner')} ({lease.get('lease_id')})"
            if i in claimed:
                why[i] += f"; {claim_label(claimed[i])}"
        elif i in running:
            why[i] = f"busy without a lease (container {running[i]['name']})"
            if i in claimed:
                why[i] += f"; {claim_label(claimed[i])}"
        elif gpu.get("used_mb", 0) > FREE_MAX_USED_MB:
            why[i] = f"busy without a lease ({gpu['used_mb']} MB in use)"
            if i in claimed:
                why[i] += f"; {claim_label(claimed[i])}"
        elif claim is None and i in claimed:
            why[i] = claim_label(claimed[i])
        elif claim is not None and claimed.get(i, {}).get("name") != claim:
            why[i] = f"not in claim {claim}"
        else:
            free.append(gpu)
    return free, why


def place(gpus_needed: list[dict[str, Any]], free: list[dict]) -> list[int] | None:
    """One card per requirement, in requirement order; None if they do not all fit."""
    pool = sorted(free, key=lambda g: (g["total_mb"], g["index"]))
    chosen: dict[int, int] = {}
    for pos, req in sorted(enumerate(gpus_needed), key=lambda r: -int(r[1]["min_vram_mb"])):
        lo, hi = int(req["min_vram_mb"]), int(req.get("max_vram_mb", 1 << 30))
        fit = next((g for g in pool if lo <= g["total_mb"] <= hi), None)
        if fit is None:
            return None
        chosen[pos] = fit["index"]
        pool.remove(fit)
    return [chosen[i] for i in range(len(gpus_needed))]


def port_conflict(port: int, obs: dict, leases: list[dict]) -> str | None:
    if str(port) in obs.get("ports", {}):
        return f"port {port} is serving {obs['ports'][str(port)]}"
    for lease in leases:
        if port in lease.get("ports", []):
            return f"port {port} is reserved by lease {lease.get('lease_id')}"
    return None
