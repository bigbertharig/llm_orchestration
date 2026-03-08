"""GPU agent core utilities mixin.

Extracted from gpu.py to isolate configuration loading, model catalog handling,
and general utility methods.
"""

import json
import os
import stat
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from gpu_constants import DEFAULT_LLM_MIN_TIER


class GPUCoreMixin:
    """Mixin providing config loading and utility methods."""

    def _effective_keep_alive(self) -> str:
        """Return a keep_alive duration for runtime compatibility."""
        raw = str(self.worker_keep_alive or "").strip()
        if not raw:
            return "30m"
        if raw == "-1":
            return "24h"
        return raw

    def _load_config(self, config_path: str) -> dict:
        if not Path(config_path).exists():
            template = Path(config_path).parent / "config.template.json"
            print(f"ERROR: Config file not found: {config_path}")
            if template.exists():
                print(f"  Copy the template and fill in your values:")
                print(f"  cp {template} {config_path}")
            else:
                print(f"  See config.template.json for the required schema.")
            sys.exit(1)
        with open(config_path) as f:
            return json.load(f)

    def _load_model_catalog(self, config_dir: Path) -> Dict[str, Any]:
        raw_path = str(self.config.get("model_catalog_path", "models.catalog.json")).strip()
        catalog_path = Path(raw_path)
        if not catalog_path.is_absolute():
            catalog_path = (config_dir / catalog_path).resolve()
        if not catalog_path.exists():
            print(f"ERROR: Model catalog not found: {catalog_path}")
            print("  Set model_catalog_path in config.json or create models.catalog.json.")
            sys.exit(1)
        try:
            with open(catalog_path, "r", encoding="utf-8") as f:
                catalog = json.load(f)
        except Exception as exc:
            print(f"ERROR: Failed to load model catalog {catalog_path}: {exc}")
            sys.exit(1)
        models = catalog.get("models", [])
        if not isinstance(models, list) or not models:
            print(f"ERROR: Model catalog {catalog_path} must contain a non-empty 'models' list.")
            sys.exit(1)
        return catalog

    def _build_model_tier_map(self, catalog: Dict[str, Any]) -> Dict[str, int]:
        tier_map: Dict[str, int] = {}
        for item in catalog.get("models", []):
            model_id = str(item.get("id", "")).strip()
            tier = item.get("tier")
            if not model_id:
                continue
            if not isinstance(tier, int) or tier < 1:
                continue
            tier_map[model_id] = int(tier)
        return tier_map

    def _build_model_meta_map(self, catalog: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        model_meta: Dict[str, Dict[str, Any]] = {}
        for item in catalog.get("models", []):
            model_id = str(item.get("id", "")).strip()
            if not model_id:
                continue
            placement = str(item.get("placement", "single_gpu") or "single_gpu").strip()
            groups: List[Dict[str, Any]] = []
            split_groups = item.get("split_groups", [])
            if isinstance(split_groups, list):
                for g in split_groups:
                    if not isinstance(g, dict):
                        continue
                    members = [str(m).strip() for m in g.get("members", []) if str(m).strip()]
                    if len(members) < 2:
                        continue
                    gid = str(g.get("id") or f"group_{'_'.join(sorted(members))}").strip()
                    try:
                        port = int(g.get("port"))
                    except Exception:
                        port = None
                    groups.append({"id": gid, "members": members, "port": port})

            # Backward-compatible pair schema: allowed_pairs:[["gpu-1","gpu-3"], ...]
            if not groups and isinstance(item.get("allowed_pairs"), list):
                for idx, pair in enumerate(item.get("allowed_pairs", []), start=1):
                    if not isinstance(pair, list):
                        continue
                    members = [str(m).strip() for m in pair if str(m).strip()]
                    if len(members) < 2:
                        continue
                    gid = f"group_{'_'.join(sorted(members))}"
                    groups.append({"id": gid, "members": members, "port": None})

            # GGUF path for llama-server backend
            gguf_path = str(item.get("gguf_path", "")).strip() or None

            model_meta[model_id] = {
                "placement": placement,
                "split_groups": groups,
                "tier": int(self.model_tier_by_id.get(model_id, DEFAULT_LLM_MIN_TIER)),
                "gguf_path": gguf_path,
            }
        return model_meta

    def _get_gpu_config(self, gpu_name: str) -> dict:
        for gpu in self.config["gpus"]:
            if gpu["name"] == gpu_name:
                return gpu
        raise ValueError(f"GPU '{gpu_name}' not found in config")

    def _verify_core_security(self):
        """Verify core/ directory is properly secured before starting.

        Checks: root ownership, no group/world-writable files, not running as root.
        Exits with error if any check fails — these are hard security requirements.
        """
        core_path = self.shared_path / "core"
        if not core_path.exists():
            self.logger.warning(f"core/ directory not found at {core_path}")
            return

        # Agents must never run as root
        if os.getuid() == 0:
            self.logger.error("SECURITY: Agents must NOT run as root.")
            sys.exit(1)

        # core/ must be root-owned
        st = core_path.stat()
        if st.st_uid != 0:
            self.logger.error(
                f"SECURITY: core/ is not root-owned (uid={st.st_uid}). "
                f"Run: sudo chown -R root:root {core_path}")
            sys.exit(1)

        # Files in core/ must not be group/world-writable
        for f in core_path.iterdir():
            if f.is_file():
                mode = f.stat().st_mode
                if mode & stat.S_IWOTH or mode & stat.S_IWGRP:
                    self.logger.error(
                        f"SECURITY: {f} is group/world-writable. "
                        f"Run: sudo chmod 644 {f}")
                    sys.exit(1)

        self.logger.info("core/ security check passed")

    def _read_json_file(self, path: Path) -> Optional[Dict[str, Any]]:
        try:
            if not path.exists():
                return None
            with open(path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            return raw if isinstance(raw, dict) else None
        except Exception:
            return None

    def _write_json_atomic(self, path: Path, payload: Dict[str, Any]):
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        os.replace(tmp_path, path)

    def _is_pid_alive(self, pid: Any) -> bool:
        try:
            pid_int = int(pid)
        except Exception:
            return False
        if pid_int <= 0:
            return False
        try:
            os.kill(pid_int, 0)
            return True
        except Exception:
            return False

    def _is_group_member(self, group: Dict[str, Any]) -> bool:
        members = [str(m).strip() for m in group.get("members", []) if str(m).strip()]
        return self.name in members
