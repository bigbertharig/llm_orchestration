"""Synthetic GGUF headers and model catalogs for the tests (no weights: the manager reads only the header)."""

from __future__ import annotations

import json
import struct

MIB = 1 << 20
Q8_0 = 8  # ggml type id: 32 values in 34 bytes


def _s(text: str) -> bytes:
    raw = text.encode()
    return struct.pack("<Q", len(raw)) + raw


def write_gguf(path, meta: dict, tensors: dict[str, int]) -> str:
    """A GGUF header with u32/str/bool-array metadata and q8_0 tensors of the given element counts."""
    out = b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(meta))
    for key, value in meta.items():
        out += _s(key)
        if isinstance(value, str):
            out += struct.pack("<I", 8) + _s(value)
        elif isinstance(value, list):
            out += struct.pack("<IIQ", 9, 7, len(value)) + bytes(int(v) for v in value)
        else:
            out += struct.pack("<II", 4, int(value))
    for name, n in tensors.items():
        out += _s(name) + struct.pack("<IQ", 1, n) + struct.pack("<IQ", Q8_0, 0)
    path.write_bytes(out)
    return str(path)


def dense(tmp_path, n_layer=10, layer_mb=100, output_mb=50, embd_mb=40, ctx_train=32768, name="m.gguf", **extra):
    per = layer_mb * MIB // 34 * 32  # q8_0 elements for layer_mb MiB
    tensors = {f"blk.{i}.attn_q.weight": per // 2 for i in range(n_layer)}
    tensors |= {f"blk.{i}.ffn_up.weight": per - per // 2 for i in range(n_layer)}
    tensors |= {"output.weight": output_mb * MIB // 34 * 32, "token_embd.weight": embd_mb * MIB // 34 * 32}
    meta = {"general.architecture": "llama", "llama.block_count": n_layer, "llama.context_length": ctx_train,
            "llama.attention.head_count": 8, "llama.attention.head_count_kv": 2,
            "llama.attention.key_length": 128, "llama.attention.value_length": 128, **extra}
    return write_gguf(tmp_path / name, meta, tensors)


def card(i, mb=6144):
    return {"index": i, "total_mb": mb, "used_mb": 0}


def write_catalog(path, models: dict[str, tuple[str, str]], profile: dict | None = None) -> str:
    """A model catalog (what tools/generate_profiles.py imports) for {model id: (status, gguf path)}."""
    prof = profile or {"runtime": {"batch_size": 64, "parallel": 1}, "extra_args": []}
    path.write_text(json.dumps({"image": "llama-runtime:test", "source_sha256": "test-source",
                                "roles": {}, "models": {
        mid: {"status": status, "aliases": [], "gguf_path": gguf,
              "profiles": {"default": json.loads(json.dumps(prof))}} for mid, (status, gguf) in models.items()}}))
    return str(path)
