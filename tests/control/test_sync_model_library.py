"""Focused checks for the benchmarking to Manager import report."""

from __future__ import annotations

import importlib.util
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
spec = importlib.util.spec_from_file_location(
    "sync_model_library", Path(__file__).resolve().parents[2] / "tools" / "sync_model_library.py")
sync = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sync)


def write_export(root: Path, recipes: dict | None = None) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    payload = {"schema": "gpu-rig-model-library/1",
               "registry": {"models": {"model-a": {}}}, "resolved_profiles": {},
               "load_recipes": {"schema_version": 1, "recipes": recipes or {}}}
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"),
                                       ensure_ascii=True).encode()).hexdigest()
    path = root / "model_library_export.json"
    path.write_text(json.dumps({"source_sha256": digest, **payload}))
    return path


def test_changed_keys_reports_add_remove_and_content_change():
    assert sync.changed_keys({"same": 1, "old": 1, "edit": 1},
                             {"same": 1, "new": 1, "edit": 2}) == {
        "added": ["new"], "removed": ["old"], "changed": ["edit"]}


def test_inspect_detects_profile_change_from_export(tmp_path, monkeypatch):
    root = tmp_path / "benchmarking"
    write_export(root)
    catalog_path = tmp_path / "model_catalog.json"
    offer_path = tmp_path / "remote_profiles.json"
    old_catalog = {"models": {"a": {"profiles": {"default": {"runtime": {"ctx_size": 4096}}}}},
                   "roles": {}, "summary": {}, "image": "llama-runtime:x",
                   "load_recipes": {"schema_version": 1, "recipes": {}}}
    catalog_path.write_text(sync.gen.render(old_catalog))
    offer = {"remote_profiles": [{"id": "a", "ctx_size": 4096}]}
    offer_path.write_text(sync.gen.render(offer))
    monkeypatch.setattr(sync.gen, "CATALOG_PATH", catalog_path)
    monkeypatch.setattr(sync.gen, "OUTPUT_PATH", offer_path)
    fresh = json.loads(json.dumps(old_catalog))
    report = sync.inspect(root, fresh, offer)
    assert not report["catalog_stale"] and not report["offers_stale"]
    fresh["models"]["a"]["profiles"]["default"]["runtime"]["ctx_size"] = 8192
    report = sync.inspect(root, fresh, {"remote_profiles": [{"id": "a", "ctx_size": 8192}]})
    assert report["catalog_stale"] and report["offers_stale"]
    assert report["models"]["changed"] == ["a"]
    assert report["offers"]["changed"] == ["a"]


def test_recipe_handoff_is_evidence_linked_and_not_executable(tmp_path):
    root = tmp_path / "benchmarking"
    recipe = {"model": "model-a", "status": "research", "procedure": "private_kv_v1",
              "placement": {"weights": ["gpu0"], "chat_state": ["gpu_workers"]},
              "evidence": ["results/runtime_experiments/case/README.md"]}
    write_export(root, {"case_v1": recipe})
    assert sync.gen.load_recipes(root, {"models": {"model-a": {}}})["recipes"]["case_v1"] == recipe

    recipe["status"] = "manager_ready"
    write_export(root, {"case_v1": recipe})
    import pytest
    with pytest.raises(ValueError, match="Manager execution requires"):
        sync.gen.load_recipes(root, {"models": {"model-a": {}}})

    recipe["status"] = "validated"
    recipe["evidence"] = ["../outside.md"]
    write_export(root, {"case_v1": recipe})
    with pytest.raises(ValueError, match="missing or invalid evidence"):
        sync.gen.load_recipes(root, {"models": {"model-a": {}}})


def test_export_hash_tampering_is_rejected(tmp_path):
    path = write_export(tmp_path)
    doc = json.loads(path.read_text())
    doc["registry"]["models"]["extra"] = {}
    path.write_text(json.dumps(doc))
    import pytest
    with pytest.raises(ValueError, match="hash mismatch"):
        sync.gen.load_export(tmp_path)


def test_atomic_text_replaces_complete_file(tmp_path):
    path = tmp_path / "catalog.json"
    path.write_text("old")
    sync._atomic_text(path, "new\n")
    assert path.read_text() == "new\n"
    assert list(tmp_path.iterdir()) == [path]


def test_sync_preserves_reviewed_subagent_candidates(tmp_path, monkeypatch):
    root = tmp_path / "benchmarking"
    write_export(root)
    candidates = {"schema_version": 1, "candidates": {"child": {"provider": "codex", "model": "gpt-test"}}}

    class Library:
        def load_registry(self, path):
            return {"models": {"model-a": {}}}

        def load_roles(self, path):
            return {"roles": {}}

        def validate(self, registry, roles):
            return []

        def generated_drift(self, registry):
            return []

    monkeypatch.setattr(sync.gen, "benchmarking_root", lambda: root)
    monkeypatch.setattr(sync.gen, "load_library", lambda root: Library())
    monkeypatch.setattr(sync.gen, "build", lambda *args: ({"remote_profiles": []}, []))
    monkeypatch.setattr(sync.gen, "load_recipes", lambda *args: {"schema_version": 1, "recipes": {}})
    monkeypatch.setattr(sync.gen, "load_subagent_catalog", lambda *args: candidates)
    monkeypatch.setattr(sync.gen, "_summary", lambda root: {})
    monkeypatch.setattr(sync.gen, "build_catalog", lambda library, registry, roles, summary, recipes, subagents:
                        {"models": {}, "roles": {}, "load_recipes": recipes, "image": "test",
                         "subagent_catalog": subagents})
    catalog_path = tmp_path / "model_catalog.json"
    monkeypatch.setattr(sync.gen, "CATALOG_PATH", catalog_path)
    monkeypatch.setattr(sync.gen, "OUTPUT_PATH", tmp_path / "remote_profiles.json")
    monkeypatch.setattr(sys, "argv", ["sync_model_library.py"])

    assert sync.main() == 0
    assert json.loads(catalog_path.read_text())["subagent_catalog"] == candidates
