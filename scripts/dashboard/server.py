"""HTTP server and request handler for the dashboard."""

import argparse
import json
import re
import shlex
import subprocess
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from .data import summarize
from .plans import (
    collect_batch_outputs,
    default_plan_config,
    discover_active_arm_bindings,
    discover_plan_input_files,
    discover_plan_starters,
    discover_plans,
    discover_recent_batches,
    plan_input_help,
    run_shell,
    write_inline_input_file,
)
from .utils import (
    INLINE_FILE_KEYS,
    load_config,
    load_json,
    normalize_github_url,
    resolve_shared_path,
    sanitize_config_object,
    sanitize_text,
)
from .workers import load_brain_state


# Template directory relative to this file
TEMPLATE_DIR = Path(__file__).parent / "templates"
BATCH_ID_RE = re.compile(r"^[A-Za-z0-9_.:-]{1,80}$")
GITHUB_URL_PARSE_RE = re.compile(r"^https://github\.com/([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+?)/?$")

# Template cache (loaded once at startup)
_template_cache: dict[str, str] = {}


def _load_template(name: str) -> str:
    """Load and cache a template file."""
    if name not in _template_cache:
        path = TEMPLATE_DIR / name
        _template_cache[name] = path.read_text(encoding="utf-8")
    return _template_cache[name]


def _render_dashboard_html() -> str:
    """Render the main dashboard HTML with inlined CSS/JS."""
    html = _load_template("dashboard.html")
    base_css = _load_template("base.css")
    dashboard_js = _load_template("dashboard.js")
    html = html.replace("{{BASE_CSS}}", base_css)
    html = html.replace("{{DASHBOARD_JS}}", dashboard_js)
    return html


def _render_controls_html() -> str:
    """Render the controls page HTML with inlined CSS/JS."""
    html = _load_template("controls.html")
    base_css = _load_template("base.css")
    controls_js = _load_template("controls.js")
    html = html.replace("{{BASE_CSS}}", base_css)
    html = html.replace("{{CONTROLS_JS}}", controls_js)
    return html


class DashboardHandler(BaseHTTPRequestHandler):
    """HTTP request handler for dashboard endpoints."""

    shared_path: Path
    config: dict[str, Any]

    def _send_json(self, data: dict[str, Any], status: int = HTTPStatus.OK) -> None:
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self, html: str) -> None:
        body = html.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json_body(self) -> dict[str, Any]:
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            length = 0
        raw = self.rfile.read(length) if length > 0 else b"{}"
        try:
            data = json.loads(raw.decode("utf-8") or "{}")
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    def _gpu_ssh(self, remote_cmd: str, timeout_s: int = 180) -> dict[str, Any]:
        # Pass remote command as a single shell-quoted argument so ssh does not
        # split `bash -lc` arguments and silently execute a different command.
        cmd = ["ssh", "-o", "BatchMode=yes", "gpu", f"bash -lc {shlex.quote(remote_cmd)}"]
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout_s,
            )
            return {
                "ok": proc.returncode == 0,
                "returncode": proc.returncode,
                "stdout": proc.stdout[-4000:],
                "stderr": proc.stderr[-4000:],
                "cmd": " ".join(cmd),
            }
        except subprocess.TimeoutExpired as e:
            return {
                "ok": False,
                "returncode": -1,
                "stdout": (e.stdout or "")[-2000:] if isinstance(e.stdout, str) else "",
                "stderr": (e.stderr or "")[-2000:] if isinstance(e.stderr, str) else "",
                "cmd": " ".join(cmd),
                "error": f"timeout after {timeout_s}s",
            }

    def _gpu_has_git(self) -> tuple[bool, str]:
        """Return whether git is available on the GPU host plus diagnostics."""
        out = self._gpu_ssh("command -v git >/dev/null 2>&1 && echo __OK__ || echo __MISSING__", timeout_s=20)
        stdout = str(out.get("stdout", ""))
        if out.get("ok") and "__OK__" in stdout:
            return True, ""
        detail = str(out.get("stderr", "")).strip() or stdout.strip() or "git not found on gpu host"
        return False, detail

    def _gpu_find_existing_repo_checkout(self, repo_url: str) -> tuple[str, str]:
        """Return (repo_path, normalized_url) for an existing local checkout on GPU host if found."""
        normalized = normalize_github_url(repo_url)
        m = GITHUB_URL_PARSE_RE.match(normalized)
        if not m:
            return "", normalized
        owner, repo = m.group(1), m.group(2)
        repo_base = f"{owner}-{repo}"
        candidates = [
            f"/mnt/shared/plans/arms/{repo_base}",
            f"/mnt/shared/plans/arms/exploratory/{repo_base}",
            f"/media/bryan/shared/plans/arms/{repo_base}",
            f"/media/bryan/shared/plans/arms/exploratory/{repo_base}",
        ]
        script_lines = ["set -e"]
        for path in candidates:
            qpath = shlex.quote(path)
            script_lines.append(f"if [ -d {qpath}/.git ]; then printf '%s\\n' {qpath}; exit 0; fi")
        script_lines.append("exit 1")
        out = self._gpu_ssh("\n".join(script_lines), timeout_s=20)
        if out.get("ok"):
            line = str(out.get("stdout", "") or "").strip().splitlines()
            if line and line[0].strip():
                return line[0].strip(), normalized
        return "", normalized

    def _discover_local_repo_choices(self) -> list[dict[str, str]]:
        """Discover git checkouts under arms/shoulders for dropdown selection."""
        plans_root = self.shared_path / "plans"
        seen: set[str] = set()
        out: list[dict[str, str]] = []

        def _to_mnt_path(p: Path) -> str:
            text = str(p)
            if text.startswith("/media/bryan/shared/"):
                return "/mnt/shared/" + text[len("/media/bryan/shared/"):]
            return text

        candidate_dirs: list[tuple[str, Path]] = [
            ("shoulders", plans_root / "shoulders"),
            ("arms", plans_root / "arms"),
            ("arms/exploratory", plans_root / "arms" / "exploratory"),
        ]

        for scope_label, base in candidate_dirs:
            try:
                if not base.exists() or not base.is_dir():
                    continue
            except Exception:
                continue
            try:
                children = sorted(base.iterdir())
            except Exception:
                continue
            for child in children:
                try:
                    if not child.is_dir():
                        continue
                    if not (child / ".git").is_dir():
                        continue
                except Exception:
                    continue
                mnt_path = _to_mnt_path(child)
                if mnt_path in seen:
                    continue
                seen.add(mnt_path)
                out.append(
                    {
                        "path": mnt_path,
                        "label": f"{scope_label}: {child.name} ({mnt_path})",
                        "scope": scope_label,
                        "name": child.name,
                    }
                )

        out.sort(key=lambda r: (str(r.get("scope", "")), str(r.get("name", ""))))
        return out

    def _control_options(self) -> dict[str, Any]:
        brain = load_brain_state(self.shared_path)
        active = brain.get("active_batches", {}) if isinstance(brain.get("active_batches"), dict) else {}
        shoulder_plans = discover_plans(self.shared_path, plan_scope="shoulders")
        arm_plans = discover_plans(self.shared_path, plan_scope="arms")
        plans = shoulder_plans + [p for p in arm_plans if p not in set(shoulder_plans)]
        plan_scopes: dict[str, str] = {}
        for p in shoulder_plans:
            plan_scopes.setdefault(p, "shoulders")
        for p in arm_plans:
            plan_scopes[p] = "arms"
        shoulder_arm_bindings = discover_active_arm_bindings(self.shared_path)
        recent_batches = discover_recent_batches(self.shared_path, active)
        plan_defaults = {p: default_plan_config(self.shared_path, p) for p in plans}
        plan_starters = {p: discover_plan_starters(self.shared_path, p, plan_scope=plan_scopes.get(p, "shoulders")) for p in plans}
        plan_input_files = {p: discover_plan_input_files(self.shared_path, p, plan_scope=plan_scopes.get(p, "shoulders")) for p in plans}
        plan_default_starter = {p: ("plan.md" if "plan.md" in plan_starters[p] else (plan_starters[p][0] if plan_starters[p] else "")) for p in plans}
        plan_inputs = {
            p: {starter: plan_input_help(self.shared_path, p, starter, plan_scope=plan_scopes.get(p, "shoulders")) for starter in plan_starters[p]}
            for p in plans
        }
        active_meta = []
        batch_labels: dict[str, str] = {}
        for batch_id, meta in active.items():
            if not isinstance(meta, dict):
                continue
            plan_name = str(meta.get("plan", "")).strip() or "unknown_plan"
            active_meta.append(
                {
                    "batch_id": batch_id,
                    "plan": plan_name,
                    "scope": "arms" if "/plans/arms/" in str(meta.get("plan_dir", "")) else "shoulders",
                    "started_at": meta.get("started_at", ""),
                }
            )
            batch_labels[batch_id] = f"{plan_name} | {batch_id}"
        for row in recent_batches:
            batch_id = str(row.get("batch_id", "")).strip()
            if not batch_id:
                continue
            plan_name = str(row.get("plan", "")).strip() or "unknown_plan"
            batch_labels.setdefault(batch_id, f"{plan_name} | {batch_id}")
        active_meta.sort(key=lambda x: str(x.get("started_at") or ""), reverse=True)
        return {
            "ok": True,
            "active_batches": sorted(active.keys()),
            "active_batches_meta": active_meta,
            "resumable_batches": sorted(active.keys()),
            "recent_batches": recent_batches,
            "plans": plans,
            "shoulder_plans": shoulder_plans,
            "arm_plans": arm_plans,
            "plan_scopes": plan_scopes,
            "shoulder_arm_bindings": shoulder_arm_bindings,
            "plan_defaults": plan_defaults,
            "plan_starters": plan_starters,
            "plan_input_files": plan_input_files,
            "plan_default_starter": plan_default_starter,
            "plan_inputs": plan_inputs,
            "batch_labels": batch_labels,
            "local_repo_choices": self._discover_local_repo_choices(),
        }

    def _kill_plan(self, batch_id: str) -> dict[str, Any]:
        if not batch_id:
            return {"ok": False, "message": "batch_id is required"}
        if not BATCH_ID_RE.fullmatch(batch_id):
            return {"ok": False, "message": f"invalid batch_id: {batch_id!r}"}
        scripts_dir = Path(__file__).resolve().parent.parent
        local_kill = scripts_dir / "kill_plan.py"
        cmd = f"python3 {shlex.quote(str(local_kill))} {shlex.quote(batch_id)} --keep-workers --keep-models --no-default-warm"
        out = run_shell(cmd, timeout_s=240)
        cleaned = self._cleanup_stale_processing_heartbeats()
        out["cleanup"] = cleaned
        if cleaned.get("errors"):
            out["stderr"] = f"{out.get('stderr', '')}\nheartbeat_cleanup_errors={cleaned.get('errors')}".strip()
        out["message"] = f"Killed batch {batch_id}" if out.get("ok") else f"Failed to kill batch {batch_id}"
        return out

    def _kill_all_active(self) -> dict[str, Any]:
        scripts_dir = Path(__file__).resolve().parent.parent
        local_kill = scripts_dir / "kill_plan.py"
        cmd = f"python3 {local_kill} --keep-workers --keep-models --no-default-warm"
        out = run_shell(cmd, timeout_s=300)
        cleaned = self._cleanup_stale_processing_heartbeats()
        out["cleanup"] = cleaned
        if cleaned.get("errors"):
            out["stderr"] = f"{out.get('stderr', '')}\nheartbeat_cleanup_errors={cleaned.get('errors')}".strip()
        out["message"] = "Killed all active batches" if out.get("ok") else "Failed to kill all active batches"
        return out

    def _cleanup_stale(self) -> dict[str, Any]:
        heartbeats = self._cleanup_stale_processing_heartbeats()
        artifacts = self._cleanup_stale_task_artifacts()
        errors = []
        if isinstance(heartbeats, dict):
            errors.extend(list(heartbeats.get("errors") or []))
        if isinstance(artifacts, dict):
            errors.extend(list(artifacts.get("errors") or []))
        ok = not errors
        if isinstance(heartbeats, dict) and isinstance(artifacts, dict):
            msg = (
                "Cleanup stale complete: "
                f"heartbeats_removed={int(heartbeats.get('removed', 0))}, "
                f"heartbeat_kept={int(heartbeats.get('kept', 0))}, "
                f"task_locks_removed={int(artifacts.get('task_locks_removed', 0))}, "
                f"failed_meta_removed={int(artifacts.get('failed_meta_removed', 0))}"
            )
        else:
            msg = "Cleanup stale complete"
        return {
            "ok": ok,
            "message": msg if ok else f"{msg} (with errors)",
            "cleanup": {
                "heartbeats": heartbeats,
                "artifacts": artifacts,
            },
            "stderr": "" if ok else f"cleanup_errors={errors}",
        }

    def _cleanup_stale_processing_heartbeats(self) -> dict[str, Any]:
        processing_dir = self.shared_path / "tasks" / "processing"
        removed = 0
        kept = 0
        errors: list[str] = []
        brain = load_brain_state(self.shared_path)
        active_batches = set()
        if isinstance(brain.get("active_batches"), dict):
            active_batches = {str(k) for k in brain.get("active_batches", {}).keys()}

        for hb in sorted(processing_dir.glob("*.heartbeat.json")):
            task_json = hb.with_name(hb.name.replace(".heartbeat.json", ".json"))
            task = load_json(task_json) if task_json.exists() else None
            should_remove = False

            if not task_json.exists() or not isinstance(task, dict):
                should_remove = True
            else:
                batch_id = str(task.get("batch_id", "")).strip()
                # Keep heartbeat only if its task still belongs to an active batch.
                if not batch_id or batch_id not in active_batches:
                    should_remove = True

            if should_remove:
                try:
                    hb.unlink()
                    removed += 1
                except Exception as exc:
                    errors.append(f"{hb.name}: {exc}")
            else:
                kept += 1

        return {"removed": removed, "kept": kept, "errors": errors}

    def _cleanup_stale_task_artifacts(self) -> dict[str, Any]:
        tasks_dir = self.shared_path / "tasks"
        errors: list[str] = []
        task_locks_removed = 0
        failed_meta_removed = 0

        brain = load_brain_state(self.shared_path)
        active_batches = set()
        if isinstance(brain.get("active_batches"), dict):
            active_batches = {str(k) for k in brain.get("active_batches", {}).keys()}

        # Remove orphan/stale task lock files. These are filesystem lock artifacts only.
        for lane in ("queue", "processing", "failed"):
            lane_dir = tasks_dir / lane
            if not lane_dir.exists():
                continue
            for lock_path in sorted(lane_dir.glob("*.json.lock")):
                task_path = lock_path.with_name(lock_path.name[:-5])  # strip ".lock"
                # Keep a lock if a task file currently exists in processing (possible active writer).
                if lane == "processing" and task_path.exists():
                    continue
                try:
                    lock_path.unlink()
                    task_locks_removed += 1
                except Exception as exc:
                    errors.append(f"{lock_path.name}: {exc}")

        # Remove stale blocked-cloud meta load failures from inactive batches; these are terminal artifacts
        # that skew dashboard failed counts after cleanup/reset.
        failed_dir = tasks_dir / "failed"
        if failed_dir.exists():
            for task_path in sorted(failed_dir.glob("*.json")):
                try:
                    task = load_json(task_path)
                except Exception as exc:
                    errors.append(f"{task_path.name}: {exc}")
                    continue
                if not isinstance(task, dict):
                    continue
                if str(task.get("status", "")).strip() != "blocked_cloud":
                    continue
                if str(task.get("task_class", "")).strip() != "meta":
                    continue
                if str(task.get("command", "")).strip() not in {"load_llm", "load_split_llm"}:
                    continue
                batch_id = str(task.get("batch_id", "")).strip()
                if batch_id and batch_id in active_batches:
                    continue
                try:
                    task_path.unlink()
                    failed_meta_removed += 1
                except Exception as exc:
                    errors.append(f"{task_path.name}: {exc}")

        return {
            "task_locks_removed": task_locks_removed,
            "failed_meta_removed": failed_meta_removed,
            "errors": errors,
        }

    def _return_default(self) -> dict[str, Any]:
        # Pass commands via stdin so pkill can't match its own shell's cmdline.
        script = (
            "pkill -f /mnt/shared/agents/brain.py || true\n"
            "pkill -f /mnt/shared/agents/gpu.py || true\n"
            "pkill -f /mnt/shared/agents/startup.py || true\n"
            "sleep 2\n"
            "nohup /home/bryan/ml-env/bin/python /mnt/shared/agents/startup.py "
            "--config /mnt/shared/agents/config.json "
            ">> /mnt/shared/logs/startup-manual.log 2>&1 < /dev/null &\n"
            "sleep 4\n"
            "pgrep -af startup.py || true\n"
            "pgrep -af brain.py || true\n"
            "pgrep -af gpu.py || true\n"
        )
        try:
            proc = subprocess.run(
                ["ssh", "-o", "BatchMode=yes", "gpu", "bash", "-s"],
                input=script,
                capture_output=True,
                text=True,
                timeout=240,
            )
            return {
                "ok": proc.returncode == 0,
                "returncode": proc.returncode,
                "stdout": proc.stdout[-4000:],
                "stderr": proc.stderr[-4000:],
                "cmd": "ssh gpu bash -s (stdin: kill agents, restart startup.py)",
                "message": "Returned system to default startup state",
            }
        except subprocess.TimeoutExpired:
            return {
                "ok": False,
                "message": "Returned system to default startup state",
                "error": "timeout after 240s",
                "cmd": "ssh gpu bash -s",
            }

    def _start_plan(
        self,
        plan_name: str,
        config_json: str,
        plan_scope: str = "shoulders",
        starter_file: str = "",
        repo_path: str = "",
        repo_url: str = "",
        claimed_behavior: str = "",
        inline_query_text: str = "",
    ) -> dict[str, Any]:
        if not plan_name:
            return {"ok": False, "message": "plan_name is required"}
        scope = str(plan_scope or "").strip().lower()
        if scope not in {"shoulders", "arms"}:
            return {"ok": False, "message": f"invalid plan_scope: {plan_scope}"}

        plans = set(discover_plans(self.shared_path, plan_scope=scope))
        if plan_name not in plans:
            return {"ok": False, "message": f"unknown plan in {scope}: {plan_name}"}

        starters = set(discover_plan_starters(self.shared_path, plan_name, plan_scope=scope))
        if starter_file and starter_file not in starters:
            return {"ok": False, "message": f"invalid starter_file for {plan_name}: {starter_file}"}

        try:
            cfg_obj = json.loads(config_json) if config_json.strip() else {}
            if not isinstance(cfg_obj, dict):
                raise ValueError("config JSON must be an object")
        except Exception as e:
            return {"ok": False, "message": f"invalid config JSON: {e}"}

        # Fill missing keys from plan defaults so required placeholders are always resolvable.
        default_cfg = default_plan_config(self.shared_path, plan_name)
        if isinstance(default_cfg, dict):
            merged_cfg = dict(default_cfg)
            merged_cfg.update(cfg_obj)
            cfg_obj = merged_cfg

        try:
            cfg_obj = sanitize_config_object(cfg_obj)
        except Exception as e:
            return {"ok": False, "message": f"invalid config values: {e}"}

        repo_path_clean = sanitize_text(repo_path, max_len=800, single_line=True) if str(repo_path).strip() else ""
        try:
            repo_url_clean = normalize_github_url(repo_url) if str(repo_url).strip() else ""
        except Exception as e:
            return {"ok": False, "message": f"invalid repo_url: {e}"}
        claimed_clean = sanitize_text(claimed_behavior, max_len=8000, single_line=True) if str(claimed_behavior).strip() else ""
        inline_clean = sanitize_text(inline_query_text, max_len=16000, single_line=False) if str(inline_query_text).strip() else ""

        auto_repo_path = ""
        if plan_name == "github_analyzer" and not repo_path_clean and repo_url_clean:
            try:
                auto_repo_path, repo_url_clean = self._gpu_find_existing_repo_checkout(repo_url_clean)
            except Exception:
                auto_repo_path = ""

        if repo_path_clean:
            cfg_obj["REPO_PATH"] = repo_path_clean
        if repo_url_clean:
            cfg_obj["REPO_URL"] = repo_url_clean
            if plan_name == "github_analyzer" and not repo_path_clean and not auto_repo_path:
                cfg_obj["REPO_PATH"] = ""
        if auto_repo_path:
            cfg_obj["REPO_PATH"] = auto_repo_path
        if claimed_clean:
            cfg_obj["CLAIMED_BEHAVIOR"] = claimed_clean
        if inline_clean:
            inline_path = write_inline_input_file(self.shared_path, plan_name, inline_clean, plan_scope=scope)
            chosen_key = next((k for k in INLINE_FILE_KEYS if k in cfg_obj), "QUERY_FILE")
            cfg_obj[chosen_key] = inline_path

        # github_analyzer resolves REPO_URL by cloning when REPO_PATH is empty.
        # Preflight this dependency so dashboard users get an immediate actionable error.
        if plan_name == "github_analyzer":
            repo_path_cfg = str(cfg_obj.get("REPO_PATH", "") or "").strip()
            repo_url_cfg = str(cfg_obj.get("REPO_URL", "") or "").strip()
            if repo_url_cfg and not repo_path_cfg:
                has_git, detail = self._gpu_has_git()
                if not has_git:
                    return {
                        "ok": False,
                        "message": (
                            "Cannot submit github_analyzer with REPO_URL because git is missing on GPU host. "
                            "Provide REPO_PATH to a local checkout, or install git on gpu host."
                        ),
                        "stderr": detail[-4000:],
                    }

        cfg_payload = json.dumps(cfg_obj, separators=(",", ":"))
        plan_path = f"/mnt/shared/plans/{scope}/{plan_name}"
        cfg_tmp = f"/tmp/dashboard_submit_{scope}_{plan_name}.json"
        starter_arg = starter_file.strip()
        starter_opt = f" --plan-file {shlex.quote(starter_arg)}" if starter_arg else ""
        script = (
            "set -e\n"
            f"cat > {cfg_tmp} <<'__CFG__'\n"
            f"{cfg_payload}\n"
            "__CFG__\n"
            f"python3 /mnt/shared/agents/submit.py {plan_path}{starter_opt} --config \"$(cat {cfg_tmp})\"\n"
            f"rm -f {cfg_tmp}\n"
        )
        try:
            proc = subprocess.run(
                ["ssh", "-o", "BatchMode=yes", "gpu", "bash", "-s"],
                input=script,
                capture_output=True,
                text=True,
                timeout=120,
            )
            out = {
                "ok": proc.returncode == 0,
                "returncode": proc.returncode,
                "stdout": proc.stdout[-4000:],
                "stderr": proc.stderr[-4000:],
                "cmd": "ssh gpu bash -s (stdin: write config JSON to temp file, run submit.py)",
            }
        except subprocess.TimeoutExpired as e:
            out = {
                "ok": False,
                "returncode": -1,
                "stdout": (e.stdout or "")[-2000:] if isinstance(e.stdout, str) else "",
                "stderr": (e.stderr or "")[-2000:] if isinstance(e.stderr, str) else "",
                "cmd": "ssh gpu bash -s",
                "error": "timeout after 120s",
            }
        display_starter = starter_arg or "plan.md"
        out["message"] = (
            f"Submitted {scope}/{plan_name} ({display_starter})"
            if out.get("ok")
            else f"Failed to submit {scope}/{plan_name} ({display_starter})"
        )
        if out.get("ok") and plan_name == "github_analyzer":
            resolved_repo = str(cfg_obj.get("REPO_PATH", "") or "").strip()
            if resolved_repo:
                out["message"] += f" [local repo: {resolved_repo}]"
        return out

    def _resume_plan(self, batch_id: str) -> dict[str, Any]:
        if not batch_id:
            return {"ok": False, "message": "batch_id is required"}
        if not BATCH_ID_RE.fullmatch(batch_id):
            return {"ok": False, "message": f"invalid batch_id: {batch_id!r}"}

        brain = load_brain_state(self.shared_path)
        active = brain.get("active_batches", {}) if isinstance(brain.get("active_batches"), dict) else {}
        meta = active.get(batch_id) if isinstance(active.get(batch_id), dict) else {}
        if not meta:
            return {"ok": False, "message": f"batch_id not found in active_batches: {batch_id}"}

        plan_name = str(meta.get("plan") or "").strip()
        if not plan_name:
            plan_dir = str(meta.get("plan_dir") or "").strip()
            if plan_dir:
                plan_name = Path(plan_dir).name
        if not plan_name:
            return {"ok": False, "message": f"could not resolve plan name for batch {batch_id}"}

        cfg = meta.get("config") if isinstance(meta.get("config"), dict) else {}
        cfg = dict(cfg)
        cfg["RUN_MODE"] = "resume"
        cfg["BATCH_ID"] = batch_id
        cfg_json = json.dumps(cfg)

        plan_scope = "shoulders"
        plan_dir = str(meta.get("plan_dir") or "").strip()
        if "/plans/arms/" in plan_dir:
            plan_scope = "arms"

        out = self._start_plan(plan_name, cfg_json, plan_scope=plan_scope)
        if out.get("ok"):
            out["message"] = f"Resumed batch {batch_id} ({plan_scope}/{plan_name})"
        else:
            out["message"] = f"Failed to resume batch {batch_id} ({plan_scope}/{plan_name})"
        return out

    def _batch_outputs(self, batch_id: str) -> dict[str, Any]:
        if not batch_id:
            return {"ok": False, "message": "batch_id is required"}
        if not BATCH_ID_RE.fullmatch(batch_id):
            return {"ok": False, "message": f"invalid batch_id: {batch_id!r}"}
        return collect_batch_outputs(self.shared_path, batch_id)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/status":
            raw_batch_ids = parse_qs(parsed.query).get("batch_ids", [""])
            selected_batch_ids: list[str] = []
            invalid_batch_ids: list[str] = []
            if raw_batch_ids:
                joined = ",".join(raw_batch_ids)
                for part in joined.split(","):
                    bid = str(part).strip()
                    if not bid:
                        continue
                    if not BATCH_ID_RE.fullmatch(bid):
                        invalid_batch_ids.append(bid)
                        continue
                    selected_batch_ids.append(bid)
            payload = summarize(self.shared_path, self.config, selected_batch_ids=selected_batch_ids)
            if invalid_batch_ids:
                alerts = payload.get("alerts")
                if not isinstance(alerts, list):
                    alerts = []
                    payload["alerts"] = alerts
                bad = ", ".join(invalid_batch_ids[:5])
                if len(invalid_batch_ids) > 5:
                    bad += ", ..."
                alerts.append(
                    {
                        "severity": "warn",
                        "worker": "dashboard",
                        "message": f"Ignored invalid tracked batch_ids: {bad}",
                    }
                )
            self._send_json(payload)
            return
        if parsed.path == "/api/control/options":
            self._send_json(self._control_options())
            return
        if parsed.path in ("/", "/index.html"):
            self._send_html(_render_dashboard_html())
            return
        if parsed.path == "/controls":
            self._send_html(_render_controls_html())
            return
        self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        payload = self._read_json_body()

        if parsed.path == "/api/control/kill_plan":
            self._send_json(self._kill_plan(str(payload.get("batch_id", "")).strip()))
            return
        if parsed.path == "/api/control/kill_all_active":
            self._send_json(self._kill_all_active())
            return
        if parsed.path == "/api/control/cleanup_stale":
            self._send_json(self._cleanup_stale())
            return
        if parsed.path == "/api/control/return_default":
            self._send_json(self._return_default())
            return
        if parsed.path == "/api/control/resume_plan":
            self._send_json(self._resume_plan(str(payload.get("batch_id", "")).strip()))
            return
        if parsed.path == "/api/control/batch_outputs":
            self._send_json(self._batch_outputs(str(payload.get("batch_id", "")).strip()))
            return
        if parsed.path == "/api/control/start_plan":
            self._send_json(
                self._start_plan(
                    str(payload.get("plan_name", "")).strip(),
                    str(payload.get("config_json", "")).strip(),
                    str(payload.get("plan_scope", "shoulders")).strip(),
                    str(payload.get("starter_file", "")).strip(),
                    str(payload.get("repo_path", "")).strip(),
                    str(payload.get("repo_url", "")).strip(),
                    str(payload.get("claimed_behavior", "")).strip(),
                    str(payload.get("inline_query_text", "")).strip(),
                )
            )
            return

        self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def log_message(self, fmt: str, *args: Any) -> None:
        return


def main() -> None:
    """Run the dashboard HTTP server."""
    parser = argparse.ArgumentParser(description="Run local orchestration dashboard")
    default_config = Path(__file__).resolve().parent.parent.parent / "shared" / "agents" / "config.json"
    parser.add_argument("--config", default=str(default_config), help="Path to config.json")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8787)
    args = parser.parse_args()

    config_path = Path(args.config)
    config = load_config(config_path)
    shared_path = resolve_shared_path(config_path, config)

    class BoundHandler(DashboardHandler):
        pass

    BoundHandler.shared_path = shared_path
    BoundHandler.config = config

    server = ThreadingHTTPServer((args.host, args.port), BoundHandler)
    print(f"Dashboard: http://{args.host}:{args.port}")
    print(f"Shared path: {shared_path}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
