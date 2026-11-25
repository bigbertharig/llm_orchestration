from __future__ import annotations

import json
import threading
from pathlib import Path


class ProfileError(RuntimeError):
    pass


class Profiles:
    """remote_profiles.json (written by tools/generate_profiles.py), re-read whenever it changes.

    The file is already resolved against the model registry: every profile carries its ctx_size,
    gguf_path and launch settings, so the gateway never imports the benchmarking repo.
    """

    def __init__(self, path: Path, *, enabled: bool = True):
        self.enabled = enabled
        self.path = path
        self._lock = threading.Lock()
        self._stamp: float | None = None
        self._profiles: dict[str, dict] = {}

    def _load(self) -> dict[str, dict]:
        items = json.loads(self.path.read_text(encoding="utf-8")).get("remote_profiles")
        if not isinstance(items, list):
            return {}
        return {str(p["id"]): p for p in items}

    def all(self) -> dict[str, dict]:
        if not self.enabled:
            return {}
        with self._lock:
            try:
                stamp = self.path.stat().st_mtime
            except OSError as exc:
                raise ProfileError(f"cannot read {self.path}: {exc}") from exc
            if stamp != self._stamp:
                try:
                    profiles = self._load()
                except (OSError, ValueError, KeyError) as exc:
                    raise ProfileError(f"cannot load profiles from {self.path}: {exc}") from exc
                if not profiles:
                    raise ProfileError(f"no remote_profiles in {self.path}")
                self._profiles = profiles
                self._stamp = stamp
            return dict(self._profiles)
