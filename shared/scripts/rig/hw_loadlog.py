"""What a llama-server load actually got, read from its startup log. Rig-side, pure (no I/O).

The controller plans a load from the GGUF (llm-gateway `manager/fit.py`); the rig checks the result
against llama.cpp's own numbers:
- llama.cpp's fit projection per device, printed before any weights load (`common_params_fit_impl`);
- layers offloaded (`load_tensors: offloaded N/M layers to GPU`): fewer than all means part of the
  model runs on the CPU, which the rig refuses;
- the context it allocated (`llama_context: n_ctx`) and the model, KV and compute buffers per device.
CUDA device numbers in the log are positions in the container's device list (the load's GPUs in order).
"""

from __future__ import annotations

import re
from typing import Any

_PROJ = re.compile(r"common_params_fit_impl:\s+- CUDA(\d+) \(.*?\):\s+(\d+) total,\s+(\d+) used,\s+(-?\d+) free")
_BUF = re.compile(r"(CUDA\d+|CPU_Mapped|CUDA_Host) (model|KV|compute) buffer size =\s+([\d.]+) MiB")
_OFFLOAD = re.compile(r"offloaded (\d+)/(\d+) layers to GPU")
_CTX = re.compile(r"llama_context: n_ctx\s+=\s+(\d+)")
_ERR = re.compile(r"( E |error|failed|out of memory|cannot meet)", re.IGNORECASE)


def parse(log: str, gpus: list[int] | None = None) -> dict[str, Any]:
    """{offloaded, layers, full_offload, n_ctx, projection[], cards[], host_mb, errors[]}."""
    gpus = list(gpus or [])
    name = lambda dev: gpus[dev] if dev < len(gpus) else dev  # noqa: E731  (device position -> GPU id)
    projection, buffers, errors = [], {}, []
    first_projection_seen = set()
    for line in log.splitlines():
        m = _PROJ.search(line)
        if m:
            dev = int(m.group(1))
            if dev not in first_projection_seen:  # the projection with the requested parameters
                first_projection_seen.add(dev)
                projection.append({"gpu": name(dev), "total_mb": int(m.group(2)), "projected_mb": int(m.group(3)),
                                   "free_mb": int(m.group(4))})
            continue
        for dev_name, kind, mib in _BUF.findall(line):
            buffers.setdefault(dev_name, {})[kind] = buffers.setdefault(dev_name, {}).get(kind, 0.0) + float(mib)
        if _ERR.search(line) and "common_params_fit_impl: (for bugs" not in line:
            errors.append(line.strip()[-300:])
    off = _OFFLOAD.search(log)
    ctxs = _CTX.findall(log)
    cards = []
    for dev_name in sorted((d for d in buffers if d.startswith("CUDA") and d != "CUDA_Host"), key=lambda d: int(d[4:])):
        b = buffers[dev_name]
        cards.append({"gpu": name(int(dev_name[4:])), "model_mb": round(b.get("model", 0)),
                      "kv_mb": round(b.get("KV", 0)), "compute_mb": round(b.get("compute", 0)),
                      "total_mb": round(sum(b.values()))})
    host = sum(v for d in ("CPU_Mapped", "CUDA_Host") for v in buffers.get(d, {}).values())
    return {
        "offloaded": int(off.group(1)) if off else None,
        "layers": int(off.group(2)) if off else None,
        "full_offload": bool(off) and off.group(1) == off.group(2),
        "n_ctx": int(ctxs[-1]) if ctxs else None,
        "projection": projection,
        "projected_short": [p for p in projection if p["free_mb"] < 0],
        "cards": cards,
        "host_mb": round(host),
        "errors": errors[-8:],
    }


def summary(info: dict[str, Any]) -> str:
    """One readable paragraph for a failed or refused load."""
    parts = []
    if info.get("projected_short"):
        parts.append("llama.cpp projects these cards short: " + ", ".join(
            f"GPU {p['gpu']} {p['projected_mb']} of {p['total_mb']} MiB ({-p['free_mb']} MiB over)"
            for p in info["projected_short"]))
    if info.get("offloaded") is not None and not info.get("full_offload"):
        parts.append(f"only {info['offloaded']}/{info['layers']} layers on the GPUs")
    if info.get("errors"):
        parts.append("log: " + " | ".join(info["errors"][-3:]))
    return "; ".join(parts) or "no fit information in the log"
