"""Will this model fit on these cards, and at what context? The harness's own check (MANAGER.md rule 8).

The controller answers before anything reaches the rig. It reads the model's GGUF header (no weights)
and builds the memory llama.cpp will allocate per card:

- weights: every `blk.N.*` tensor goes to the card holding layer N; `token_embd` stays in host RAM
  (llama.cpp keeps the input layer on the CPU); `output` (or the tied embedding) goes to the last card;
  next-token-prediction layers (`nextn_predict_layers`) are skipped by llama.cpp and cost nothing.
- KV cache (q8_0, run_runtime.sh `-ctk/-ctv q8_0`): per attention layer, KV heads x (key + value
  length) x context. Hybrid models (`full_attention_interval`) keep KV only on their attention layers,
  recurrent layers hold a small fixed state; sliding-window layers hold at most the window.
- compute buffer per card: from this GGUF's measured loads at the same micro-batch and card count
  when available (`compute_mb`, padded), else context x micro-batch x heads as a fallback (2.7x the measured buffer on qwen3.6:35b-a3b).
- a depth-calibrated margin for matching model/card layouts, otherwise 512 MiB.

Layers are split in order across the cards in CUDA order (lowest GPU first), the output layer counting
against the last card. The plan is the largest context that fits (capped at the model's trained
context or the request), with the layers then balanced so every card has about the same headroom. The
result is exact llama.cpp arguments: `--ctx-size` and `--tensor-split` as whole layer counts (llama.cpp
assigns layer i to the card whose cumulative share first exceeds i / (layers + 1)).

Checked 2026-10-18 against llama.cpp b11669 on Ornith-1.5-9B over two 1060s: weights 4812 MiB and
KV 17.0 KiB per token predicted, the same as llama.cpp measured. The rig still runs llama.cpp's own
projection at load and refuses a load that would not fit, so this check can be conservative.
"""

from __future__ import annotations

import os
import re
import struct
from dataclasses import dataclass, field
from typing import Any, BinaryIO

from . import observations

MIB = 1 << 20
KV_BYTES_PER_ELEM = 34 / 32  # q8_0: 32 values in 34 bytes
CTX_STEP = 256               # llama.cpp pads the context to 256
CTX_MIN_DEFAULT = 4096
MARGIN_MB_DEFAULT = -1       # use depth-calibrated growth where available; 512 MiB otherwise
MARGIN_MB_FALLBACK = 512
MARGIN_MB_MIN = 128           # protects against sparse probe sampling and allocator variation
COMPUTE_BASE_MB = 64         # compute buffer at tiny context (measured 39 MiB on Ornith-9B)
CUDA_CONTEXT_MB = 150        # per card per process (CUDA context, cuBLAS workspace), not in llama.cpp's sums
PLACEMENT_POLICIES = ("smallest_fit", "fastest_fit")

# ggml type id -> (elements per block, bytes per block). Unknown ids count as 4 bytes per element.
GGML_BLOCKS = {
    0: (1, 4), 1: (1, 2), 2: (32, 18), 3: (32, 20), 6: (32, 22), 7: (32, 24), 8: (32, 34), 9: (32, 36),
    10: (256, 84), 11: (256, 110), 12: (256, 144), 13: (256, 176), 14: (256, 210), 15: (256, 292),
    16: (256, 66), 17: (256, 74), 18: (256, 98), 19: (256, 50), 20: (32, 18), 21: (256, 110),
    22: (256, 82), 23: (256, 136), 24: (1, 1), 25: (1, 2), 26: (1, 4), 27: (1, 8), 28: (1, 8),
    29: (256, 56), 30: (1, 2), 34: (256, 54), 35: (256, 66), 39: (32, 17),
}
_SCALARS = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}


class FitError(Exception):
    """The GGUF could not be read."""


# ---------------------------------------------------------------- GGUF header

def _read(f: BinaryIO, fmt: str):
    size = struct.calcsize("<" + fmt)
    raw = f.read(size)
    if len(raw) != size:
        raise FitError("truncated GGUF header")
    return struct.unpack("<" + fmt, raw)[0]


def _string(f: BinaryIO) -> str:
    return f.read(_read(f, "Q")).decode("utf-8", "replace")


def _value(f: BinaryIO, vtype: int) -> Any:
    if vtype == 8:
        return _string(f)
    if vtype == 9:
        etype, n = _read(f, "I"), _read(f, "Q")
        if etype == 8:
            for _ in range(n):
                _string(f)
            return None  # string arrays (the vocabulary) are not needed
        if etype not in _SCALARS:
            raise FitError(f"unsupported GGUF array type {etype}")
        fmt = _SCALARS[etype]
        if n > 4096:  # big numeric arrays (token scores): skip
            f.seek(n * struct.calcsize("<" + fmt), os.SEEK_CUR)
            return None
        return [_read(f, fmt) for _ in range(n)]
    if vtype not in _SCALARS:
        raise FitError(f"unsupported GGUF value type {vtype}")
    return _read(f, _SCALARS[vtype])


def read_header(path: str) -> tuple[dict[str, Any], dict[str, int]]:
    """(metadata, tensor name -> bytes) from a GGUF file, reading only the header."""
    with open(path, "rb") as f:
        if f.read(4) != b"GGUF":
            raise FitError(f"{path} is not a GGUF file")
        _read(f, "I")  # version
        n_tensors, n_kv = _read(f, "Q"), _read(f, "Q")
        meta = {}
        for _ in range(n_kv):
            key = _string(f)
            meta[key] = _value(f, _read(f, "I"))
        tensors = {}
        for _ in range(n_tensors):
            name = _string(f)
            dims = [_read(f, "Q") for _ in range(_read(f, "I"))]
            ttype = _read(f, "I")
            _read(f, "Q")  # offset
            n = 1
            for d in dims:
                n *= d
            per_block, block_bytes = GGML_BLOCKS.get(ttype, (1, 4))
            tensors[name] = -(-n // per_block) * block_bytes
    return meta, tensors


# ---------------------------------------------------------------- model shape

@dataclass
class Shape:
    """Per-layer memory facts for one GGUF, independent of the cards."""
    model: str
    arch: str
    n_ctx_train: int
    n_head: int
    weights: list[int]            # bytes per layer slot (0 for skipped layers)
    kv_per_token: list[float]     # KV bytes per context token per layer (0 for non-attention layers)
    kv_window: list[int | None]   # sliding window per layer (None: full context)
    fixed_state: list[int]        # recurrent state bytes per layer (independent of context)
    output_bytes: int             # output layer, on the last card
    host_bytes: int               # token embedding, kept in host RAM
    notes: list[str] = field(default_factory=list)
    source: str = ""              # GGUF file name: the key measured loads are calibrated by

    @property
    def n_layers(self) -> int:
        return len(self.weights)

    @property
    def weights_mb(self) -> float:
        return (sum(self.weights) + self.output_bytes) / MIB


def _per_layer(value: Any, n: int, default: Any) -> list:
    if isinstance(value, list):
        return (list(value) + [value[-1] if value else default] * n)[:n]
    return [default if value is None else value] * n


def shape_from_header(meta: dict[str, Any], tensors: dict[str, int], model: str = "") -> Shape:
    arch = meta.get("general.architecture") or ""
    k = lambda key, default=None: meta.get(f"{arch}.{key}", default)  # noqa: E731
    n_layer = int(k("block_count", 0))
    if n_layer <= 0:
        raise FitError(f"{model}: GGUF has no {arch}.block_count")
    notes = []
    nextn = int(k("nextn_predict_layers", 0) or 0)
    skipped = set(range(n_layer - nextn, n_layer)) if nextn else set()
    if skipped:
        notes.append(f"{nextn} next-token-prediction layer(s) not loaded")

    weights = [0] * n_layer
    attn_layers = set()
    for name, size in tensors.items():
        m = re.match(r"blk\.(\d+)\.", name)
        if not m:
            continue
        il = int(m.group(1))
        if il < n_layer and il not in skipped:
            weights[il] += size
            if ".attn_k." in name or ".attn_q." in name or ".attn_qkv." in name:
                attn_layers.add(il)

    n_head = int(_per_layer(k("attention.head_count", 1), 1, 1)[0] or 1)
    head_kv = _per_layer(k("attention.head_count_kv"), n_layer, n_head)
    emb = int(k("embedding_length", 0) or 0)
    key_len = int(k("attention.key_length", 0) or (emb // max(n_head, 1)))
    val_len = int(k("attention.value_length", 0) or key_len)

    interval = k("full_attention_interval")
    if interval:  # hybrid: KV only on every interval-th layer, recurrent state elsewhere
        is_attn = [(il + 1) % int(interval) == 0 for il in range(n_layer)]
        notes.append(f"hybrid: full attention every {interval} layers")
    elif attn_layers:
        is_attn = [il in attn_layers for il in range(n_layer)]
    else:
        is_attn = [True] * n_layer
    kv_per_token = [0.0] * n_layer
    for il in range(n_layer):
        if is_attn[il] and il not in skipped:
            kv_per_token[il] = float(head_kv[il] or 0) * (key_len + val_len) * KV_BYTES_PER_ELEM

    window = k("attention.sliding_window")
    pattern = k("attention.sliding_window_pattern")
    kv_window: list[int | None] = [None] * n_layer
    if window:
        if isinstance(pattern, list):
            swa = [bool(p) for p in _per_layer(pattern, n_layer, False)]
        elif pattern:
            swa = [(il + 1) % int(pattern) != 0 for il in range(n_layer)]
        else:
            swa = [False] * n_layer  # window without a pattern: treat as full context (conservative)
        kv_window = [int(window) if s else None for s in swa]
        if any(swa):
            notes.append(f"sliding window {window} on {sum(swa)} layers")

    fixed = [0] * n_layer
    inner, state = int(k("ssm.inner_size", 0) or 0), int(k("ssm.state_size", 0) or 0)
    if inner and state:
        conv, groups = int(k("ssm.conv_kernel", 4) or 4), int(k("ssm.group_count", 1) or 1)
        per = (conv - 1) * (inner + 2 * groups * state) * 4 + inner * state * 4
        fixed = [0 if (is_attn[il] or il in skipped) else per for il in range(n_layer)]

    output = tensors.get("output.weight")
    if output is None:
        output = tensors.get("token_embd.weight", 0)
        notes.append("tied embeddings: output layer reuses token_embd")
    output += sum(v for n, v in tensors.items() if n.startswith("output_norm") or n == "output.bias")
    return Shape(model=model, arch=arch, n_ctx_train=int(k("context_length", 0) or 0), n_head=n_head,
                 weights=weights, kv_per_token=kv_per_token, kv_window=kv_window, fixed_state=fixed,
                 output_bytes=int(output), host_bytes=int(tensors.get("token_embd.weight", 0)), notes=notes)


_SHAPES: dict[tuple[str, float], Shape] = {}


def shape_for(path: str, model: str = "") -> Shape:
    """Shape of a GGUF, cached per file and mtime (the header of a big vocabulary takes a moment)."""
    try:
        key = (path, os.path.getmtime(path))
    except OSError as exc:
        raise FitError(f"{model or path}: cannot read {path}: {exc}") from exc
    if key not in _SHAPES:
        meta, tensors = read_header(path)
        _SHAPES[key] = shape_from_header(meta, tensors, model)
        _SHAPES[key].source = os.path.basename(path)
    return _SHAPES[key]


# ---------------------------------------------------------------- calibration from measured loads

# Every ready load is kept in the manager's observation log (observations.py, kind `load`) with what llama.cpp
# allocated per card (the rig's `fit`, hw_loadlog.py), and the compute buffer is planned from those
# measurements for the same GGUF and micro-batch. The formula below (context x micro-batch x heads) is the
# ceiling: on qwen3.6:35b-a3b it gave 762 MiB per card at 89K against 280 measured, 2.7x, so five 1060s were
# refused 131K that they hold (2026-10-25).
COMPUTE_PAD_MB = 64       # added to a calibrated compute buffer
COMPUTE_PAD_FRAC = 0.10   # and this share of it
_CALIBRATION: dict[str, Any] = {"mtime": None, "points": {}}


def record_observation(source: str, model: str, ubatch: int, gpus: list[int], measured: dict[str, Any] | None,
                       **extra: Any) -> dict[str, Any] | None:
    """Keep one measured load (the rig's `fit`: n_ctx and buffers per card), with any other facts about it
    (`extra`: use, cards by UUID and link, the estimate, load time). Return the exact
    saved row so Gateway can forward it to rig-owned calibration."""
    cards = (measured or {}).get("cards") or []
    n_ctx = (measured or {}).get("n_ctx")
    if not source or not n_ctx or not cards or not all(c.get("compute_mb") is not None for c in cards):
        return None
    return observations.record("load", source=source, model=model, ubatch=int(ubatch), n_ctx=int(n_ctx),
                               gpus=list(gpus), cards=[{k: c.get(k) for k in ("gpu", "model_mb", "kv_mb", "compute_mb",
                                                                              "total_mb")} for c in cards], **extra)


def _points() -> dict[tuple[str, int], list[tuple[int, float]]]:
    """(GGUF, ubatch) -> [(ctx, largest compute buffer on any card)], re-read when the log changes."""
    def stamp(p) -> tuple:
        try:
            st = os.stat(p)
            return (str(p), st.st_mtime_ns, st.st_size)  # size too: two appends can share an mtime tick
        except OSError:
            return (str(p), None, None)

    mtime = stamp(observations.path())
    if _CALIBRATION["mtime"] != mtime:
        points: dict[tuple[str, int, int], list[tuple[int, float]]] = {}
        overheads: dict[int, float] = {}
        for row in observations.history("load"):
            if row.get("placement") == "ram":
                continue  # partial offload is not a full-GPU calibration
            try:
                worst = max(float(c["compute_mb"]) for c in row["cards"])
                points.setdefault((row["source"], int(row["ubatch"]), len(row["cards"])), []).append((int(row["n_ctx"]), worst))
            except (ValueError, KeyError, TypeError):
                continue
            # used_mb: nvidia-smi per card once the load served; hw: the card's size (both from the rig).
            size = {int(h["index"]): h.get("total_mb") for h in row.get("hw") or [] if h.get("index") is not None}
            used = {int(g): u for g, u in zip(row.get("gpus") or [], row.get("used_mb") or [])}
            for c in row["cards"]:
                g = int(c.get("gpu", -1))
                if size.get(g) and used.get(g) is not None and c.get("total_mb") is not None:
                    extra = float(used[g]) - float(c["total_mb"])
                    overheads[int(size[g])] = max(overheads.get(int(size[g]), 0.0), extra)
        _CALIBRATION.update(mtime=mtime, points=points, overheads=overheads)
    return _CALIBRATION["points"]


def overhead_mb(card_mb: int) -> float:
    """Memory a card loses beyond llama.cpp's own buffers (CUDA context, cuBLAS workspace, CUDA graphs): the
    most measured on a card of this size (nvidia-smi in use after a load minus llama.cpp's total), padded;
    CUDA_CONTEXT_MB until a load of that card size has been measured. The 3090 ran out of memory in
    cudaGraphInstantiate with 150 + 512 MiB planned (qwen3-coder:30b-a3b at 120K, 2026-10-25)."""
    seen = _overheads().get(int(card_mb))
    return max(CUDA_CONTEXT_MB, seen + COMPUTE_PAD_MB) if seen is not None else CUDA_CONTEXT_MB


def _overheads() -> dict[int, float]:
    _points()  # refreshes the cache
    return _CALIBRATION.get("overheads", {})


def failed_ceiling(s: Shape, ubatch: int, card_sizes: list[int]) -> int | None:
    """The smallest context at which this GGUF ran out of memory on cards of these sizes: a plan stays below it
    until a load at or above it succeeds."""
    sizes = sorted(int(c) for c in card_sizes)
    fails = [int(r["planned_ctx"]) for r in observations.rows("load_failed")
             if r.get("placement") != "ram"
             and r.get("source") == s.source and int(r.get("ubatch") or 0) == int(ubatch) and r.get("oom")
             and sorted(int(c) for c in r.get("card_sizes") or []) == sizes and r.get("planned_ctx")]
    if not fails:
        return None
    lowest = min(fails)
    # Only a success on cards of the same sizes lifts it: a bigger context on five 1060s says nothing of the 3090.
    ok = [r for r in observations.history("load")
          if r.get("placement") != "ram"
          and r.get("source") == s.source and int(r.get("ubatch") or 0) == int(ubatch)
          and int(r.get("n_ctx") or 0) >= lowest
          and sorted(int(h.get("total_mb") or 0) for h in r.get("hw") or []) == sizes]
    return None if ok else lowest


def _formula_mb(s: Shape, ctx: int, ubatch: int) -> float:
    return COMPUTE_BASE_MB + ctx * ubatch * s.n_head * 4 / MIB


def compute_basis(s: Shape, ubatch: int, n_cards: int = 1) -> str:
    n = len(_points().get((s.source, int(ubatch), n_cards), []))
    return f"measured ({n} load{'s' if n != 1 else ''}, {n_cards} card{'s' if n_cards != 1 else ''})" if n else "formula (no matched card-count load yet)"


def growth_margin_mb(s: Shape, cards: list[dict[str, Any]]) -> int:
    """Use measured full-depth growth only for the same GGUF and card-size layout.

    A sparse sampling envelope gets a fourfold allowance and a 128 MiB floor.
    Other layouts retain the original 512 MiB margin until depth-tested.
    """
    sizes = sorted(int(c["total_mb"]) for c in cards)
    growth = [max(float(v) for v in row["growth_mb"])
              for row in observations.rows("depth_memory")
              if row.get("source") == s.source
              and sorted(int(v) for v in row.get("card_sizes") or []) == sizes
              and row.get("growth_mb") and len(row["growth_mb"]) == len(cards)]
    return max(MARGIN_MB_MIN, int(max(growth) * 4 + 0.999)) if growth else MARGIN_MB_FALLBACK


# ---------------------------------------------------------------- the plan

def _layer_bytes(s: Shape, il: int, ctx: int) -> float:
    tokens = ctx if s.kv_window[il] is None else min(ctx, s.kv_window[il])
    return s.weights[il] + s.kv_per_token[il] * tokens + s.fixed_state[il]


def compute_mb(s: Shape, ctx: int, ubatch: int, n_cards: int = 1) -> float:
    """Compute buffer per card. From measured loads of this GGUF at this micro-batch when there are any:
    the line through them (one load: its measurement scaled like the formula), padded and
    never below a measured buffer at or under ctx. The formula is used for unmeasured card counts. Otherwise the formula."""
    formula = _formula_mb(s, ctx, ubatch)
    pts = sorted(_points().get((s.source, int(ubatch), n_cards), []))
    if not pts:
        return formula
    if len({c for c, _ in pts}) >= 2:
        n = len(pts)
        mx, my = sum(c for c, _ in pts) / n, sum(m for _, m in pts) / n
        slope = max(0.0, sum((c - mx) * (m - my) for c, m in pts) / sum((c - mx) ** 2 for c, _ in pts))
        est = my + slope * (ctx - mx) + max(m - (my + slope * (c - mx)) for c, m in pts)  # worst residual
    else:
        c0, m0 = pts[0]
        est = m0 * _formula_mb(s, ctx, ubatch) / _formula_mb(s, c0, ubatch)
    below = [m for c, m in pts if c <= ctx]
    est = max(est, max(below) if below else 0.0)
    # A measured buffer can exceed the old formula. Never plan below what actually loaded.
    return max((max(below) + COMPUTE_PAD_MB) if below else 0.0,
               min(formula, est * (1 + COMPUTE_PAD_FRAC) + COMPUTE_PAD_MB))


def _partition(s: Shape, ctx: int, budgets: list[float]) -> list[int] | None:
    """Contiguous layer counts per card that stay within each card's byte budget, filling cards in order
    (optimal for feasibility); the last card also carries the output layer. None if it cannot."""
    counts, il, n = [], 0, s.n_layers
    for d, budget in enumerate(budgets):
        room = budget - (s.output_bytes if d == len(budgets) - 1 else 0)
        if room < 0:
            return None
        start = il
        while il < n and _layer_bytes(s, il, ctx) <= room:
            room -= _layer_bytes(s, il, ctx)
            il += 1
        counts.append(il - start)
    return counts if il == n else None


def _card_use(s: Shape, ctx: int, counts: list[int]) -> list[dict[str, float]]:
    out, il = [], 0
    for d, c in enumerate(counts):
        layers = range(il, il + c)
        il += c
        w = sum(s.weights[i] for i in layers) + (s.output_bytes if d == len(counts) - 1 else 0)
        kv = sum(_layer_bytes(s, i, ctx) - s.weights[i] for i in layers)
        out.append({"layers": c, "weights_mb": w / MIB, "kv_mb": kv / MIB})
    return out


def plan(s: Shape, cards: list[dict[str, Any]], *, ctx: int | None = None, ctx_min: int = CTX_MIN_DEFAULT,
         ctx_max: int | None = None, ubatch: int = 64, parallel: int = 1,
         margin_mb: int = MARGIN_MB_DEFAULT, runtime_image: str | None = None) -> dict[str, Any]:
    """Fit `s` on `cards` ([{index, total_mb, used_mb}], any order; CUDA order is by index).

    ctx: an exact context that must fit; None: the largest that fits between ctx_min and ctx_max
    (default: the model's trained context). Returns {fits, ctx, tensor_split, cards[], reason, ...}.
    """
    cards = sorted(cards, key=lambda c: c["index"])
    if margin_mb < 0:
        margin_mb = growth_margin_mb(s, cards)
    n_cards = len(cards)
    gpus = [int(c["index"]) for c in cards]
    free = [float(c["total_mb"]) - float(c.get("used_mb", 0)) for c in cards]
    over = [overhead_mb(int(c["total_mb"])) for c in cards]
    top = min(ctx_max or s.n_ctx_train or 131072, s.n_ctx_train or 1 << 30)
    # A context that ran out of memory on cards like these caps the largest-that-fits plan (not an exact request).
    ceiling = failed_ceiling(s, ubatch, [int(c["total_mb"]) for c in cards])
    if ceiling and not ctx:
        top = min(top, ceiling - CTX_STEP)
    want_lo, want_hi = (ctx, ctx) if ctx else (ctx_min, max(ctx_min, top))
    want_lo = max(CTX_STEP, want_lo // CTX_STEP * CTX_STEP)
    want_hi = max(want_lo, want_hi // CTX_STEP * CTX_STEP)
    if ceiling and ctx and want_lo >= ceiling:
        return {"model": s.model, "gpus": gpus, "fits": False, "ctx": want_lo,
                "failed_ceiling": ceiling, "margin_mb": margin_mb,
                "reason": f"won't fit: {s.model} previously ran out of memory at ctx {ceiling} on these card sizes"}

    def budgets(c: int, scale: float = 1.0) -> list[float]:
        return [(f - margin_mb - o - compute_mb(s, c, ubatch, n_cards)) * MIB * scale for f, o in zip(free, over)]

    base = {"model": s.model, "arch": s.arch, "gpus": gpus, "n_ctx_train": s.n_ctx_train,
            "weights_mb": round(s.weights_mb), "host_mb": round(s.host_bytes / MIB), "margin_mb": margin_mb,
            "parallel": parallel, "notes": s.notes, "compute_basis": compute_basis(s, ubatch, n_cards),
            "overhead_mb": [round(o) for o in over], **({"failed_ceiling": ceiling} if ceiling else {})}
    if not cards:
        return {**base, "fits": False, "reason": "no cards given"}
    if _partition(s, want_lo, budgets(want_lo)) is None:
        need = s.weights_mb + sum(s.kv_per_token[i] * (want_lo if s.kv_window[i] is None
                                                       else min(want_lo, s.kv_window[i])) + s.fixed_state[i]
                                  for i in range(s.n_layers)) / MIB
        have = sum(free) - sum(over) - len(cards) * (margin_mb + compute_mb(s, want_lo, ubatch, n_cards))
        why = (f"won't fit: {s.model} needs about {need:.0f} MiB on the cards at ctx {want_lo} "
               f"(weights {s.weights_mb:.0f} + KV/state {need - s.weights_mb:.0f}); "
               f"{len(cards)} card(s) {gpus} give {max(have, 0):.0f} MiB after margins")
        if have >= need:
            why += " in total, but the layers do not divide across the cards (a layer or the output is too big)"
        return {**base, "fits": False, "ctx": want_lo, "need_mb": round(need), "have_mb": round(have),
                "reason": why}
    lo, hi = want_lo, want_hi  # largest ctx that fits, by bisection on CTX_STEP multiples
    while lo < hi:
        mid = (lo + hi + CTX_STEP) // 2 // CTX_STEP * CTX_STEP
        if _partition(s, mid, budgets(mid)) is not None:
            lo = mid
        else:
            hi = mid - CTX_STEP
    best = lo
    # Balance: the smallest uniform share of each budget that still fits, so every card keeps headroom.
    scale_lo, scale_hi = 0.0, 1.0
    for _ in range(30):
        mid = (scale_lo + scale_hi) / 2
        if _partition(s, best, budgets(best, mid)) is None:
            scale_lo = mid
        else:
            scale_hi = mid
    blocks = _partition(s, best, budgets(best, scale_hi))
    split = blocks[:-1] + [blocks[-1] + 1]  # the output layer is the last slot llama.cpp splits
    use = _card_use(s, best, blocks)
    rows = []
    for c, f, u, o in zip(cards, free, use, over):
        comp = compute_mb(s, best, ubatch, n_cards)
        total = u["weights_mb"] + u["kv_mb"] + comp + o
        rows.append({"gpu": int(c["index"]), "total_mb": int(c["total_mb"]), "free_mb": round(f),
                     "layers": u["layers"], "weights_mb": round(u["weights_mb"]), "kv_mb": round(u["kv_mb"]),
                     "compute_mb": round(comp), "projected_mb": round(total), "headroom_mb": round(f - total)})
    capped = "requested" if ctx else ("a failed load" if ceiling and best >= top else
                                      "trained context" if best >= top else "card memory")
    return {**base, "fits": True, "ctx": best, "ctx_limit": capped, "per_slot_ctx": best // max(parallel, 1),
            "tensor_split": ",".join(str(c) for c in split) if len(split) > 1 else None, "cards": rows,
            "reason": f"fits: {s.model} on GPU {gpus} at ctx {best} (limited by {capped})"}



# Host offload is an interactive fallback only. Reserve 4 GiB for the host and
# leave 2 GiB inside the rig's 20g llama container for buffers and mmap pages.
HOST_RESERVE_MB = 4096
HOST_CONTAINER_BUDGET_MB = 18 * 1024


def plan_ram(s: Shape, card: dict[str, Any], *, ctx: int, ubatch: int,
             mem_available_mb: int, margin_mb: int = MARGIN_MB_DEFAULT) -> dict[str, Any]:
    """Largest GPU suffix that fits this card; earlier layers and output stay on CPU.

    llama.cpp offloads the final N transformer layers for a numeric -ngl N. A
    partial plan keeps the exact requested context and never counts swap as RAM.
    """
    idx = int(card["index"])
    if margin_mb < 0:
        margin_mb = growth_margin_mb(s, [card])
    # Partial loads have less calibration than GPU-only loads. A Nemotron
    # 32K live load failed allocating a 611 MiB compute buffer at the old
    # 512 MiB margin (2026-11-03); keep 2 GiB free when planning CPU spill.
    margin_mb = max(margin_mb, 2048)
    gpu_budget = (float(card["total_mb"]) - float(card.get("used_mb", 0))
                  - overhead_mb(int(card["total_mb"])) - margin_mb
                  - compute_mb(s, ctx, ubatch, 1)) * MIB
    host_budget_mb = min(max(0, int(mem_available_mb) - HOST_RESERVE_MB), HOST_CONTAINER_BUDGET_MB)
    base = {"model": s.model, "arch": s.arch, "gpus": [idx], "ctx": ctx,
            "fits": False, "placement": "ram", "host_budget_mb": host_budget_mb,
            "margin_mb": margin_mb, "tensor_split": None}
    if gpu_budget <= 0 or host_budget_mb <= 0:
        return {**base, "reason": "insufficient GPU or reserved host RAM"}
    # CPU holds the input and output tensors. Count them even for tied embeddings:
    # this deliberately overestimates shared storage.
    active_layers = max((i + 1 for i, w in enumerate(s.weights) if w), default=0)
    for n_gpu in range(active_layers - 1, 0, -1):
        first = active_layers - n_gpu
        gpu_bytes = sum(_layer_bytes(s, i, ctx) for i in range(first, active_layers))
        host_bytes = (s.host_bytes + s.output_bytes
                      + sum(_layer_bytes(s, i, ctx) for i in range(first)))
        host_mb = host_bytes / MIB
        if gpu_bytes <= gpu_budget and host_mb + 2048 <= host_budget_mb:
            gpu_mb = gpu_bytes / MIB + compute_mb(s, ctx, ubatch, 1) + overhead_mb(int(card["total_mb"]))
            return {**base, "fits": True, "n_gpu_layers": n_gpu,
                    "host_estimate_mb": round(host_mb + 2048),
                    "cards": [{"gpu": idx, "total_mb": int(card["total_mb"]),
                               "free_mb": round(float(card["total_mb"]) - float(card.get("used_mb", 0))),
                               "layers": n_gpu, "projected_mb": round(gpu_mb),
                               "headroom_mb": round(float(card["total_mb"]) - float(card.get("used_mb", 0)) - gpu_mb)}],
                    "reason": f"fits: {s.model} at ctx {ctx} on GPU {idx} plus host RAM"}
    return {**base, "reason": f"host RAM or GPU budget cannot hold {s.model} at ctx {ctx}"}

def choose_cards(free_cards: list[dict[str, Any]], n: int) -> list[dict[str, Any]]:
    """n cards for a request that only gives a count: smallest cards first, so big cards stay free."""
    return sorted(free_cards, key=lambda g: (g["total_mb"], g["index"]))[:n]


def candidate_sets(free_cards: list[dict[str, Any]], n: int) -> list[list[dict[str, Any]]]:
    """Card sets of size n to try, in preference order: the n smallest (big cards stay free), then the
    biggest free card with the n-1 smallest others (one big card often beats several small ones)."""
    by_size = sorted(free_cards, key=lambda g: (g["total_mb"], g["index"]))
    sets = [by_size[:n]] if len(by_size) >= n else []
    if by_size and n <= len(by_size):
        big = by_size[-1]
        alt = [big] + [g for g in by_size if g is not big][:n - 1]
        if sorted(g["index"] for g in alt) != sorted(g["index"] for g in sets[0]):
            sets.append(alt)
    return sets


# Fallback when comparable measured speeds are not yet available. Splits cost speed on this rig.
SAME_CLASS_MAX_CARDS = 2


def _measured_placement(s: Shape, free_cards: list[dict[str, Any]], **kw: Any) -> dict[str, Any] | None:
    """Choose the lowest estimated request time only among comparable measured card sets.

    A 1024-token prompt plus 256-token answer represents an interactive request.
    Require two measured, currently free candidates for the same GGUF and micro-batch.
    An unknown placement retains the established fallback policy.
    """
    ctx, ubatch = kw.get("ctx"), int(kw.get("ubatch", 64))
    runtime_image = kw.get("runtime_image")
    if not ctx:
        return None
    pool = {int(g["index"]): g for g in free_cards}
    candidates = []
    seen = set()
    for row in observations.rows("performance"):
        if row.get("source") != s.source or int(row.get("ubatch") or 0) != ubatch:
            continue
        if runtime_image and row.get("runtime_image") != runtime_image:
            continue
        measured_ctx = int(row.get("n_ctx") or 0)
        if not measured_ctx or not (0.5 <= ctx / measured_ctx <= 2.0):
            continue
        gpus = tuple(sorted(int(g) for g in row.get("gpus") or []))
        if not gpus or gpus in seen or not all(g in pool for g in gpus):
            continue
        prefill, decode = float(row.get("prefill_tps") or 0), float(row.get("decode_tps") or 0)
        if prefill <= 0 or decode <= 0:
            continue
        candidate = plan(s, [pool[g] for g in gpus], **kw)
        if not candidate["fits"]:
            continue
        seen.add(gpus)
        latency = 1024 / prefill + 256 / decode
        candidates.append((latency, candidate, row))
    if len(candidates) < 2:
        return None
    latency, result, row = min(candidates, key=lambda x: (x[0], len(x[1]["gpus"])))
    result["performance_basis"] = row.get("record", "gateway performance observation")
    result["estimated_response_seconds"] = round(latency, 2)
    return result


def choose_smallest(s: Shape, free_cards: list[dict[str, Any]], **kw: Any) -> dict[str, Any] | None:
    """Strict smallest-fit placement, independent of measured speed."""
    classes: dict[int, list[dict[str, Any]]] = {}
    for g in free_cards:
        classes.setdefault(int(g["total_mb"]), []).append(g)
    for size in sorted(classes):
        group = sorted(classes[size], key=lambda g: g["index"])
        for n in range(1, min(SAME_CLASS_MAX_CARDS, len(group)) + 1):
            p = plan(s, group[:n], **kw)
            if p["fits"]:
                return p
    for n in range(1, len(free_cards) + 1):
        for cards in candidate_sets(free_cards, n):
            p = plan(s, cards, **kw)
            if p["fits"]:
                return p
    return None


def choose(s: Shape, free_cards: list[dict[str, Any]], **kw: Any) -> dict[str, Any] | None:
    """Measured fastest placement when comparable evidence exists; otherwise strict smallest-fit."""
    measured = _measured_placement(s, free_cards, **kw)
    return measured if measured is not None else choose_smallest(s, free_cards, **kw)


def choose_min(s: Shape, free_cards: list[dict[str, Any]], **kw: Any) -> dict[str, Any] | None:
    """Fewest cards that hold the profile minimum; prefer smaller cards within that count."""
    for n in range(1, len(free_cards) + 1):
        for cards in candidate_sets(free_cards, n):
            p = plan(s, cards, **kw)
            if p["fits"]:
                return p
    return None


def choose_chat(s: Shape, free_cards: list[dict[str, Any]], **kw: Any) -> dict[str, Any] | None:
    """Interactive placement: largest card, then its smallest fitting GPU split.

    On this rig that means 3090-only, then 3090 + the fewest 1060s. If the
    3090 is unavailable, try the 1060s alone. Explicit GPU requests bypass
    this choice. Host-RAM and Strata offload are separate runtime paths.
    """
    if not free_cards:
        return None
    by_size = sorted(free_cards, key=lambda g: (-int(g["total_mb"]), int(g["index"])))
    primary = by_size[0]
    p = plan(s, [primary], **kw)
    if p["fits"]:
        return p
    others = sorted(by_size[1:], key=lambda g: (int(g["index"])))
    for n in range(1, len(others) + 1):
        p = plan(s, [primary, *others[:n]], **kw)
        if p["fits"]:
            return p
    return choose(s, free_cards, **kw)


def suggest(s: Shape, free_cards: list[dict[str, Any]], **kw: Any) -> list[str]:
    """Alternatives for a request that does not fit: fewest small cards, or one card that holds it."""
    out = []
    for n in range(1, len(free_cards) + 1):
        pick = choose_cards(free_cards, n)
        p = plan(s, pick, **kw)
        if p["fits"]:
            out.append(f"{n} card(s) GPU {p['gpus']}: ctx {p['ctx']}")
            break
    for c in sorted(free_cards, key=lambda g: -g["total_mb"])[:1]:
        p = plan(s, [c], **kw)
        if p["fits"] and f"GPU {[c['index']]}" not in " ".join(out):
            out.append(f"1 card GPU {c['index']} ({c['total_mb']} MB): ctx {p['ctx']}")
    return out
