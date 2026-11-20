"""Rig-owned, read-only model fit against the imported library and live cards."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import fit


PROFILE_BY_USE = {"benchmark": "default", "chat": "interactive", "worker": "orchestration"}


class ModelFitError(Exception):
    def __init__(self, kind: str, message: str, **details: Any):
        super().__init__(message)
        self.kind = kind
        self.details = details


def _int(request: dict, name: str, default: int | None = None) -> int | None:
    value = request.get(name, default)
    if value is not None and (type(value) is not int or value < 0):
        raise ModelFitError("invalid", f"{name} must be a nonnegative integer")
    return value


def _selected_model(library: Any, request: dict) -> tuple[dict, dict]:
    snapshot = library.load_snapshot()
    registry = snapshot["registry"]
    model = request.get("model")
    use = request.get("use", "benchmark")
    if not isinstance(model, str) or not model:
        raise ModelFitError("invalid", "model is required")
    if use not in PROFILE_BY_USE:
        raise ModelFitError("invalid", "use must be chat, worker, or benchmark")
    canonical = model if model in registry["models"] else next(
        (mid for mid, entry in registry["models"].items() if model in entry.get("aliases", [])), None)
    if canonical is None:
        raise ModelFitError("not_found", f"unknown model {model}")
    entry = registry["models"][canonical]
    if use != "benchmark" and (entry.get("status") != "active" or entry.get("in_testing")):
        raise ModelFitError("not_found", f"model {canonical} is not approved for routine runtime use")
    profile = request.get("profile") or PROFILE_BY_USE[use]
    if profile not in entry["profiles"]:
        if request.get("profile"):
            raise ModelFitError("not_found", f"model {canonical} has no profile {profile}")
        profile = "default"
    return snapshot, {**library.resolve(canonical, profile, registry),
                      "model_size_bytes": entry.get("model_size_bytes"),
                      "ngram_size_bytes": entry.get("ngram_size_bytes", 0),
                      "strata": entry.get("strata"), "loading_stage": entry.get("loading_stage")}


def _context(r: dict, request: dict, train: int | None) -> tuple[int | None, int, int]:
    mode = request.get("context_mode", "working")
    if mode not in ("working", "min", "max"):
        raise ModelFitError("invalid", "context_mode must be working, min, or max")
    ctx = _int(request, "ctx")
    if ctx is not None and mode != "working":
        raise ModelFitError("invalid", "ctx is exact; context_mode applies only when ctx is omitted")
    bounds = r.get("context_bounds") or {}
    lower = max(int(bounds.get("min") or fit.CTX_MIN_DEFAULT), _int(request, "ctx_min", fit.CTX_MIN_DEFAULT))
    upper = min(int(bounds.get("max") or train or 131072), train or 1 << 30)
    tested = int((r.get("loading_stage") or {}).get("context_limit") or 0)
    if tested:
        upper = min(upper, tested)
    if lower > upper:
        raise ModelFitError("invalid", f"context minimum {lower} exceeds model maximum {upper}")
    if ctx is not None:
        if ctx > upper or ctx < lower:
            raise ModelFitError("invalid", f"ctx {ctx} is outside profile range {lower}..{upper}")
    elif mode == "min":
        ctx = lower
    elif mode == "working" and (r.get("runtime") or {}).get("ctx_size"):
        ctx = max(lower, int(r["runtime"]["ctx_size"]))
        if ctx > upper:
            raise ModelFitError("invalid", f"working ctx {ctx} exceeds model maximum {upper}")
    return ctx, lower, upper


def _strata(r: dict, request: dict, free: list[dict], host_available: int | None) -> dict:
    cfg = r.get("strata") or {}
    ctx, minimum, maximum = _context(r, request, int(cfg.get("max_context") or 0))
    wanted = (ctx if ctx is not None else maximum if request.get("context_mode") == "max" else
              minimum if request.get("context_mode") == "min" else
              max(int((r.get("runtime") or {}).get("ctx_size") or minimum), minimum))
    ids = request.get("gpu_ids")
    validated = cfg.get("validated_gpu_ids") or []
    selected = next((g for g in free if g["index"] in validated), None)
    reason = (f"{r['model']} has no registered Strata pack" if not cfg else
              "no Benchmarking-validated Strata GPU placement" if not validated else
              f"validated Strata placement uses GPU {validated} alone" if
              request.get("cards") not in (None, 1) or
              (ids is not None and ids != [selected["index"]] if selected else ids is not None) else
              f"Strata context {wanted} is outside validated range {minimum}..{maximum}" if
              wanted < minimum or wanted > maximum else
              f"validated Strata GPU {validated} is unavailable" if selected is None else
              f"Strata needs {cfg.get('min_host_available_mb')} MiB available host RAM" if
              host_available is None or host_available < int(cfg.get("min_host_available_mb") or 0) else None)
    return {"fits": reason is None, "reason": reason or "registered Strata pack and resources available",
            "placement": "strata", "backend": "strata",
            "gpus": [selected["index"]] if selected else [], "ctx": wanted,
            "tensor_split": None, "cards": [], "strata": cfg,
            "model_size_bytes": r.get("model_size_bytes"),
            "ngram_size_bytes": r.get("ngram_size_bytes", 0),
            "total_size_bytes": (r["model_size_bytes"] + r.get("ngram_size_bytes", 0))
            if r.get("model_size_bytes") else None,
            "chosen_by": "rig: registered Strata tier"}


def preview(request: dict, *, library: Any, cards: list[dict], free: list[dict],
            unavailable: dict[int, str], host_available: int | None) -> dict:
    """Return a fit proposal; lease acquisition still rechecks admission atomically."""
    if not isinstance(request, dict):
        raise ModelFitError("invalid", "model fit request must be an object")
    count = _int(request, "cards")
    ids = request.get("gpu_ids")
    if ids is not None and (not isinstance(ids, list) or not ids or
                            any(type(g) is not int or g < 0 for g in ids) or len(ids) != len(set(ids))):
        raise ModelFitError("invalid", "gpu_ids must be a nonempty list of distinct card indices")
    if count and ids:
        raise ModelFitError("invalid", "give cards or gpu_ids, not both")
    snapshot, r = _selected_model(library, request)
    policy = request.get("placement_policy") or ("fastest_fit" if request.get("use") == "chat" else "smallest_fit")
    if policy not in fit.PLACEMENT_POLICIES:
        raise ModelFitError("invalid", f"placement_policy must be one of {fit.PLACEMENT_POLICIES}")
    engine = request.get("engine", "auto")
    if engine not in ("auto", "llama", "strata"):
        raise ModelFitError("invalid", "engine must be auto, llama, or strata")
    if engine == "llama" and (r.get("strata") or {}).get("only"):
        raise ModelFitError("invalid", f"{r['model']} is a Strata-only pack")
    if engine == "strata" or (engine == "auto" and (r.get("strata") or {}).get("only")):
        result = _strata(r, request, free, host_available)
    else:
        try:
            shape = fit.shape_for(str(Path(r["gguf_path"])), r["model"])
        except fit.FitError as exc:
            raise ModelFitError("not_found", str(exc)) from exc
        ctx, lower, upper = _context(r, request, shape.n_ctx_train)
        rt = r.get("runtime") or {}
        kw = {"ctx": ctx, "ctx_min": lower, "ctx_max": upper,
              "ubatch": min(int(_int(request, "batch_size") or rt.get("batch_size", 64)), 512),
              "parallel": int(rt.get("parallel", 1)),
              "margin_mb": request.get("margin_mb", fit.MARGIN_MB_DEFAULT),
              "runtime_image": snapshot["registry"].get("runtime_image")}
        mode = request.get("context_mode", "working")
        if type(kw["margin_mb"]) is not int or kw["margin_mb"] < fit.MARGIN_MB_DEFAULT:
            raise ModelFitError("invalid", "margin_mb must be -1 or a nonnegative integer")
        if ids:
            blocked = {g: unavailable.get(g, "not present") for g in ids if g not in {f["index"] for f in free}}
            if blocked:
                raise ModelFitError("conflict", "GPU(s) unavailable: " + "; ".join(
                    f"GPU {g} {why}" for g, why in blocked.items()), unavailable=blocked)
            chosen = [g for g in free if g["index"] in ids]
        elif count:
            if count > len(free):
                raise ModelFitError("capacity", f"{count} card(s) requested; {len(free)} free", unavailable=unavailable)
            sets = fit.candidate_sets(free, count)
            chosen = (max(sets, key=lambda group: fit.plan(shape, group, **kw).get("ctx", -1)) if mode == "max" else
                      next((group for group in sets if fit.plan(shape, group, **kw)["fits"]), sets[0]))
        else:
            chosen = None
        if chosen is not None:
            result = fit.plan(shape, chosen, **kw)
        else:
            fastest = policy == "fastest_fit"
            result = (fit.choose_chat(shape, free, **kw) if mode == "max" and fastest else
                      fit.plan(shape, free, **kw) if mode == "max" else
                      fit.choose_min(shape, free, **kw) if mode == "min" else
                      fit.choose_chat(shape, free, **kw) if fastest else
                      fit.choose_smallest(shape, free, **kw))
            result = result or fit.plan(shape, fit.choose_cards(free, len(free)), **kw)
        if not result.get("fits") and request.get("use") == "chat" and chosen is None and ctx is not None:
            if host_available is not None:
                ordering = (lambda g: (-g["total_mb"], g["index"])) if policy == "fastest_fit" else (
                    lambda g: (g["total_mb"], g["index"]))
                for card in sorted(free, key=ordering):
                    ram = fit.plan_ram(shape, card, ctx=ctx, ubatch=kw["ubatch"],
                                       mem_available_mb=host_available, margin_mb=kw["margin_mb"])
                    if ram["fits"]:
                        result = ram
                        break
        if not result.get("fits") and engine == "auto" and r.get("strata"):
            strata = _strata(r, request, free, host_available)
            if strata["fits"]:
                result = strata
        result.update(ubatch=kw["ubatch"], context_mode=mode,
                      context_bounds={"min": lower, "working": rt.get("ctx_size"), "max": upper,
                                      "max_basis": (r.get("context_bounds") or {}).get("max_basis", "gguf_train")})
    result.update(registry_profile=r["profile"], status=r["status"], use=request.get("use", "benchmark"),
                  gguf_path=r["gguf_path"], placement_policy=policy,
                  source_sha256=snapshot["source_sha256"])
    if not result["fits"]:
        raise ModelFitError("capacity", result["reason"], fit=result, unavailable=unavailable)
    return result
