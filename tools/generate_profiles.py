#!/usr/bin/env python3
"""Import the benchmarking model library into the Orchestration's own files: the only place the Orchestration reads it.

The model registry (benchmarking repo, model_registry.json) is where benchmarking keeps what it has learned
about how each model loads: GGUF, context, batch, extra args, per named profile. This importer copies that
into files the Orchestration owns, so the Orchestration runs with no benchmarking checkout:

- model_catalog.json: every registry model and its resolved profiles, the roles, the runtime image, and the
  dashboard's model summary (scores, runtime). The manager plans and loads from it (manager/models.py).
- remote_profiles.json: the offer, the models the Orchestration publishes on /v1/models under which registry
  profile. The offer names no cards: the manager's fit places a published profile and sizes its context.

The Orchestration service runs it at startup when the Benchmarking export is available.

Usage (with Benchmarking in a sibling checkout or BENCHMARKING_ROOT set):
  tools/generate_profiles.py            rewrite both files
  tools/generate_profiles.py --check    exit 1 if the offer is invalid or either file is stale
  tools/generate_profiles.py --sync-if-available  service startup; use the last valid snapshot if source is unavailable
  BENCHMARKING_ROOT=<path> ...          a different benchmarking checkout
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
OUTPUT_PATH = REPO / "remote_profiles.json"
CATALOG_PATH = REPO / "model_catalog.json"
BENCHMARKING_ROOTS = ()
# Routine runtime offers are limited to reviewed models. Candidates remain in the
# catalog for explicit benchmark loads, including orchestration test procedures.
OFFERABLE_STATUSES = {"active"}
USES = ("chat", "worker", "benchmark")  # manager/memory.py USES
RECIPE_STATUSES = {"research", "validated"}
RECIPE_TIERS = {"gpu0", "gpu_workers", "host_ram", "nvme"}
RECIPE_COMPONENTS = {"weights", "experts", "lookup", "chat_state"}

# The offer: what the Orchestration publishes, under which registry profile. Model settings come from that profile,
# never from this list, and so do card choice and context: the manager's fit places each published profile
# on the cards it fits best and plans the largest context that fits, at least the profile's ctx_size.
# use (chat | worker | benchmark, default chat) sizes the host prompt cache (manager/memory.py).
OFFER: list[dict[str, Any]] = [
    {"id": "chat-27b", "model": "qwen3.6:27b", "profile": "interactive", "default": True,
     "description": "Strong general/coding model for interactive agents (Roo)."},
    {"id": "coder-7b", "model": "qwen2.5-coder:7b", "profile": "interactive", "default": True,
     "description": "Fast small coder for scoped/bulk tasks. No native tool calls."},
    {"id": "coder-30b", "model": "qwen3-coder:30b-a3b", "profile": "interactive", "default": False,
     "description": "Coding MoE (30B, 3B active). Roo candidate."},
    {"id": "devstral-24b", "model": "devstral-small:24b", "profile": "interactive", "default": False,
     "description": "Agentic coding model (built for tool-using agents). Roo candidate."},
    {"id": "gpt-oss-20b", "model": "gpt-oss:20b", "profile": "interactive", "default": False,
     "description": "MoE (21B, 3.6B active), strong tool calling, fast. Roo candidate."},
    {"id": "harness-7b", "model": "qwen2.5-coder:7b", "profile": "interactive", "default": False,
     "description": "Separate 7B parent route for harness size comparisons."},
    {"id": "harness-14b", "model": "qwen2.5-coder:14b", "profile": "orchestration", "default": False,
     "description": "14B parent route for harness size comparisons."},
    {"id": "harness-gemma4-e4b-64k", "model": "gemma-4:e4b", "profile": "default", "default": False,
     "ctx_size": 65536, "description": "One GTX 1060 parent route for harness comparison."},
    {"id": "harness-gemma4-12b-64k", "model": "gemma-4:12b", "profile": "default", "default": False,
     "ctx_size": 65536, "description": "Two GTX 1060 parent route for harness comparison."},
    {"id": "harness-ministral-8b", "model": "ministral-3:8b", "profile": "default", "default": False,
     "description": "Alternate small-model parent route for harness tool comparisons."},
    {"id": "harness-ministral-3b", "model": "ministral-3:3b", "profile": "default", "default": False,
     "description": "Compact single-GPU route for harness tool comparisons."},
    {"id": "harness-ornith-9b", "model": "ornith-1.5:9b", "profile": "default", "default": False,
     "description": "Alternate small-model parent route for harness tool comparisons."},
    {"id": "harness-phi-14b", "model": "phi-4:14b", "profile": "default", "default": False,
     "description": "Alternate 14B parent route for harness tool comparisons."},
    {"id": "harness-74b", "model": "nemotron-3.5-bolt:74b-a9b", "profile": "default", "default": False,
     "description": "74B parent route for harness size comparisons."},
    {"id": "harness-qwen38-27b", "model": "qwen3.8:27b", "profile": "interactive", "default": False,
     "description": "Validated Qwen 3.8 27B route for supervised Harness trials."},
]

GENERATED_NOTE = ("Generated by tools/generate_profiles.py from the model registry. Do not edit; "
                  "change OFFER in the script or the registry profile, then re-run it.")
CATALOG_NOTE = ("Imported by tools/generate_profiles.py from the benchmarking model library. Do not edit; "
                "the Orchestration reads models only from here. Re-run the importer to take in new lessons.")


def benchmarking_root(*, optional: bool = False) -> Path | None:
    env = os.getenv("BENCHMARKING_ROOT")
    candidates = [Path(env)] if env else list(BENCHMARKING_ROOTS)
    for root in candidates:
        if (root / "model_library_export.json").is_file():
            return root
    if optional:
        return None
    raise SystemExit(f"model library export not found under {[str(c) for c in candidates]}; set BENCHMARKING_ROOT")


def verify_local_snapshot() -> str:
    """Use the deployed snapshot only when Benchmarking is not installed here."""
    catalog = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    profiles = json.loads(OUTPUT_PATH.read_text(encoding="utf-8"))
    digest = catalog.get("source_sha256")
    if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ValueError("local catalog has no Benchmarking source hash")
    if not isinstance(catalog.get("models"), dict) or not catalog["models"]:
        raise ValueError("local model catalog is empty")
    if not isinstance(profiles.get("remote_profiles"), list):
        raise ValueError("local remote profiles are missing")
    if profiles.get("source_sha256") != digest:
        raise ValueError("local offer and catalog have different source hashes")
    for offer in profiles["remote_profiles"]:
        if offer.get("model") not in catalog["models"]:
            raise ValueError(f"local offer names unknown model: {offer.get('model')}")
    return digest


def load_export(root: Path) -> dict[str, Any]:
    path = root / "model_library_export.json"
    doc = json.loads(path.read_text(encoding="utf-8"))
    if doc.get("schema") != "gpu-rig-model-library/1":
        raise ValueError("unsupported benchmarking model library export")
    payload = {key: value for key, value in doc.items() if key != "source_sha256"}
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"),
                                       ensure_ascii=True).encode("utf-8")).hexdigest()
    if digest != doc.get("source_sha256"):
        raise ValueError("benchmarking model library export hash mismatch")
    if not isinstance(doc.get("registry", {}).get("models"), dict) or not isinstance(doc.get("resolved_profiles"), dict):
        raise ValueError("benchmarking model library export is incomplete")
    return doc


class ExportLibrary:
    def __init__(self, doc: dict[str, Any]):
        self.doc = doc

    def load_registry(self, _path: Path) -> dict:
        return self.doc["registry"]

    def load_roles(self, _path: Path) -> dict:
        return self.doc["roles"]

    def resolve(self, model: str, profile: str, _registry: dict) -> dict:
        if model in self.doc["resolved_profiles"]:
            return self.doc["resolved_profiles"][model][profile]
        for canonical, entry in self.doc["registry"]["models"].items():
            if model in entry.get("aliases", []):
                return self.doc["resolved_profiles"][canonical][profile]
        raise KeyError(model)

    def runtime_image(self, _registry: dict) -> str:
        return self.doc["registry"]["runtime_image"]

    def validate(self, registry: dict, roles: dict) -> list[str]:
        models = registry["models"]
        return [f"role {name} names an unknown model" for name, spec in roles["roles"].items()
                if spec.get("model") not in models]

    def generated_drift(self, _registry: dict) -> list[str]:
        return []


def load_library(root: Path) -> ExportLibrary:
    return ExportLibrary(load_export(root))


def load_recipes(root: Path, registry: dict[str, Any]) -> dict[str, Any]:
    """Validate exported research recipes. Manager never executes these IDs directly."""
    doc = load_export(root)["load_recipes"]
    if (not isinstance(doc, dict) or set(doc) != {"schema_version", "recipes"}
            or type(doc["schema_version"]) is not int or doc["schema_version"] != 1):
        raise ValueError("load_recipes.json: expected schema_version 1 and recipes")
    recipes = doc["recipes"]
    if not isinstance(recipes, dict):
        raise ValueError("load_recipes.json: recipes must be an object")
    for rid, recipe in recipes.items():
        if not isinstance(rid, str) or not re.fullmatch(r"[a-z][a-z0-9_]*", rid):
            raise ValueError(f"load_recipes.json: invalid recipe id {rid!r}")
        if not isinstance(recipe, dict) or set(recipe) != {
                "model", "status", "procedure", "placement", "evidence"}:
            raise ValueError(f"load_recipes.json: {rid}: invalid fields")
        if not isinstance(recipe["model"], str) or recipe["model"] not in registry["models"]:
            raise ValueError(f"load_recipes.json: {rid}: unknown model {recipe['model']!r}")
        if not isinstance(recipe["status"], str) or recipe["status"] not in RECIPE_STATUSES:
            raise ValueError(f"load_recipes.json: {rid}: status must be research or validated; "
                             "Manager execution requires a separate capability implementation")
        if not isinstance(recipe["procedure"], str) or not re.fullmatch(
                r"[a-z][a-z0-9_]*", recipe["procedure"]):
            raise ValueError(f"load_recipes.json: {rid}: procedure must be an identifier, not a command")
        placement = recipe["placement"]
        if not isinstance(placement, dict) or not placement or set(placement) - RECIPE_COMPONENTS:
            raise ValueError(f"load_recipes.json: {rid}: invalid placement components")
        for component, tiers in placement.items():
            if (not isinstance(tiers, list) or not tiers
                    or any(not isinstance(t, str) or t not in RECIPE_TIERS for t in tiers)):
                raise ValueError(f"load_recipes.json: {rid}: invalid tiers for {component}")
        evidence = recipe["evidence"]
        if not isinstance(evidence, list) or not evidence:
            raise ValueError(f"load_recipes.json: {rid}: evidence must be a nonempty list")
        for ref in evidence:
            if (not isinstance(ref, str) or not ref.startswith("results/runtime_experiments/")
                or not ref.endswith(".md") or ".." in Path(ref).parts):
                raise ValueError(f"load_recipes: {rid}: missing or invalid evidence {ref!r}")
    return doc


def build(offer: list[dict[str, Any]], library: Any, registry: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """(remote_profiles document, problems). Profiles with problems are left out."""
    problems: list[str] = []
    profiles = []
    seen = set()
    for item in offer:
        pid = item["id"]
        if pid in seen:
            problems.append(f"{pid}: offered twice")
            continue
        seen.add(pid)
        before = len(problems)
        try:
            r = library.resolve(item["model"], item.get("profile", "default"), registry)
        except KeyError as exc:
            problems.append(f"{pid}: {exc.args[0]}")
            continue
        runtime = r.get("runtime", {})
        if r["status"] not in OFFERABLE_STATUSES or registry["models"][r["model"]].get("in_testing"):
            problems.append(f"{pid}: {r['model']} is not approved for routine runtime offers")
        if "ctx_size" not in runtime:
            problems.append(f"{pid}: registry profile {r['profile']} lacks runtime ctx_size")
        if "gpus_needed" in item or "tensor_split" in item:
            problems.append(f"{pid}: the offer names no cards; the manager's fit places it")
        if item.get("use", "chat") not in USES:
            problems.append(f"{pid}: use must be one of {USES}")
        if len(problems) > before:
            continue
        profiles.append({
            "id": pid,
            "model": r["model"],
            "registry_profile": r["profile"],
            "status": r["status"],
            "description": item.get("description", ""),
            "default": bool(item.get("default", False)),
            "good_for": r.get("good_for", []),
            "dashboard_chat": "chatbot" in r.get("good_for", []),  # compatibility for running Orchestration
            "use": item.get("use", "chat"),
            "ctx_size": item.get("ctx_size", runtime["ctx_size"]),  # minimum context for this route
        })
    doc = {"_comment": GENERATED_NOTE, "registry_updated": registry.get("updated"),
           "source_sha256": getattr(library, "doc", {}).get("source_sha256"),
           "remote_profiles": profiles}
    return doc, problems


def load_subagent_catalog(root: Path, registry: dict[str, Any]) -> dict[str, Any]:
    """Copy reviewed child identities, never readiness or a rig command."""
    doc = load_export(root)["subagent_catalog"]
    if doc.get("schema_version") != 1 or not isinstance(doc.get("candidates"), dict):
        raise ValueError("subagent_catalog.json: expected schema_version 1 and candidates")
    for name, row in doc["candidates"].items():
        if not re.fullmatch(r"[a-z][a-z0-9_]*", name) or not isinstance(row, dict):
            raise ValueError(f"subagent_catalog.json: invalid candidate {name!r}")
        if row.get("provider") not in ("rig_shared_chat", "gateway_profile", "codex") or not isinstance(row.get("model"), str):
            raise ValueError(f"subagent_catalog.json: invalid provider or model for {name}")
        if not isinstance(row.get("agent_types"), list) or not row["agent_types"] or any(
                role not in ("coding", "research") for role in row["agent_types"]):
            raise ValueError(f"subagent_catalog.json: invalid agent_types for {name}")
        if row["provider"] == "rig_shared_chat":
            requires = row.get("requires")
            if (row["model"] not in registry["models"]
                    or not isinstance(row.get("mode"), str)
                    or not re.fullmatch(r"gpu[a-z0-9-]+", row["mode"])
                    or not isinstance(requires, dict)
                    or requires.get("model") != row["model"]
                    or not isinstance(requires.get("mode"), str)
                    or not re.fullmatch(r"gpu[a-z0-9-]+", requires["mode"])):
                raise ValueError(f"subagent_catalog.json: invalid shared chat mode for {name}")
            if type(row.get("context_cap")) is not int or row["context_cap"] <= 0:
                raise ValueError(f"subagent_catalog.json: invalid context cap for {name}")
            if row.get("context_allocation") not in ("fixed_per_chat", "dynamic_at_attach"):
                raise ValueError(f"subagent_catalog.json: invalid context allocation for {name}")
            if row.get("context_allocation") == "dynamic_at_attach" and (
                    type(row.get("context_min")) is not int or row["context_min"] < 1
                    or type(row.get("context_default")) is not int
                    or not row["context_min"] <= row["context_default"] <= row["context_cap"]):
                raise ValueError(f"subagent_catalog.json: invalid dynamic context range for {name}")
            if type(row.get("max_attached_chats")) is not int or row["max_attached_chats"] <= 0:
                raise ValueError(f"subagent_catalog.json: invalid attached chat limit for {name}")
        elif row["provider"] == "gateway_profile":
            if row["model"] not in registry["models"]:
                raise ValueError(f"subagent_catalog.json: unknown Orchestration profile model for {name}")
        elif not row["model"].startswith("gpt-") or row.get("availability") != "check_codex_model_list":
            raise ValueError(f"subagent_catalog.json: invalid Codex candidate {name}")
    return doc


def build_catalog(library: Any, registry: dict[str, Any], roles: dict[str, Any] | None,
                  summary: dict[str, Any] | None,
                  load_recipes: dict[str, Any] | None = None,
                  subagent_catalog: dict[str, Any] | None = None) -> dict[str, Any]:
    """The Orchestration's copy of the model library: every model with its resolved profiles, the roles, the image,
    and the dashboard's model summary."""
    models = {}
    for mid, entry in registry["models"].items():
        if not isinstance(entry, dict):
            continue
        profiles, gguf = {}, None
        for name in entry.get("profiles") or {}:
            try:
                r = library.resolve(mid, name, registry)
            except KeyError:
                continue
            gguf = gguf or r.get("gguf_path")
            profiles[name] = {k: v for k, v in r.items() if k not in ("model", "profile", "status", "gguf_path")}
        model_size = entry.get("model_size_bytes")
        ngram_size = entry.get("ngram_size_bytes", 0)
        if model_size is not None:
            if type(model_size) is not int or model_size <= 0 or type(ngram_size) is not int or ngram_size < 0:
                raise ValueError(f"{mid}: invalid model_size_bytes or ngram_size_bytes")
        models[mid] = {"status": entry.get("status"),
                       **({"in_testing": True} if entry.get("in_testing") else {}),
                       **({"loading_stage": entry["loading_stage"]} if "loading_stage" in entry else {}),
                       "aliases": entry.get("aliases", []), "gguf_path": gguf,
                       "model_size_bytes": model_size, "ngram_size_bytes": ngram_size,
                       "total_size_bytes": model_size + ngram_size if model_size is not None else None,
                       "strata": entry.get("strata"), "profiles": profiles}
    role_rows = {name: {k: spec.get(k) for k in ("model", "profile", "lane", "why", "good_for")}
                 for name, spec in ((roles or {}).get("roles") or {}).items() if isinstance(spec, dict)}
    return {"_comment": CATALOG_NOTE,
            "source_sha256": getattr(library, "doc", {}).get("source_sha256"),
            "registry_updated": registry.get("updated"),
            "roles_updated": (roles or {}).get("updated"), "image": library.runtime_image(registry),
            "models": models, "roles": role_rows, "summary": summary,
            "load_recipes": load_recipes or {"schema_version": 1, "recipes": {}},
            "subagent_catalog": subagent_catalog or {"schema_version": 1, "candidates": {}}}


def render(doc: dict[str, Any]) -> str:
    return json.dumps(doc, indent=2, ensure_ascii=True) + "\n"


def _summary(root: Path) -> dict[str, Any] | None:
    """The benchmark-owned summary included in the portable export."""
    return load_export(root)["summary"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--check", action="store_true", help="validate and compare, do not write")
    ap.add_argument("--sync-if-available", action="store_true",
                    help="import Benchmarking at startup; use local snapshot if source is absent")
    args = ap.parse_args()
    if args.check and args.sync_if_available:
        ap.error("--check and --sync-if-available are mutually exclusive")

    root = benchmarking_root(optional=args.sync_if_available)
    if root is None:
        print(f"Benchmarking unavailable; using Orchestration snapshot {verify_local_snapshot()}")
        return 0
    library = load_library(root)
    registry = library.load_registry(root / "model_registry.json")
    doc, problems = build(OFFER, library, registry)
    roles = library.load_roles(root / "model_roles.json")
    try:
        recipes = load_recipes(root, registry)
    except (OSError, ValueError) as exc:
        problems.append(str(exc))
        recipes = None
    try:
        subagents = load_subagent_catalog(root, registry)
    except (OSError, ValueError) as exc:
        problems.append(str(exc))
        subagents = None
    catalog = build_catalog(library, registry, roles, _summary(root), recipes, subagents)
    for p in problems:
        print(f"problem: {p}")
    outputs = {OUTPUT_PATH: render(doc), CATALOG_PATH: render(catalog)}
    if args.check:
        for path, text in outputs.items():
            current = path.read_text(encoding="utf-8") if path.is_file() else ""
            # the summary carries its build time; compare without it
            strip = (lambda t: re.sub(r'"generated_at": "[^"]*"', "", t)) if path == CATALOG_PATH else (lambda t: t)
            if strip(current) != strip(text):
                problems.append(f"{path.name} stale")
                print(f"problem: {path.name} differs from the model library; run tools/generate_profiles.py")
        print("ok" if not problems else f"{len(problems)} problem(s)")
        return 1 if problems else 0
    if problems:
        print(f"not written: {len(problems)} problem(s)", file=sys.stderr)
        return 1
    for path, text in outputs.items():
        if not path.is_file() or path.read_text(encoding="utf-8") != text:
            path.write_text(text, encoding="utf-8", newline="\n")
    print(f"Benchmarking source {library.doc['source_sha256']} from {root}")
    print(f"wrote {OUTPUT_PATH.name}: {', '.join(p['id'] for p in doc['remote_profiles'])}")
    print(f"wrote {CATALOG_PATH.name}: {len(catalog['models'])} models, {len(catalog['roles'])} roles")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError, TypeError) as exc:
        if "--sync-if-available" not in sys.argv:
            raise
        print(f"Benchmarking update unavailable: {exc}", file=sys.stderr)
        print(f"Using Orchestration snapshot {verify_local_snapshot()}")
