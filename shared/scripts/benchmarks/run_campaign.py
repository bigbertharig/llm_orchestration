#!/usr/bin/env python3
"""Unified benchmark campaign runner.

Reads a declarative manifest (model + suite + settings per block) and manages the full lifecycle:
load model -> run suite -> reap -> advance. Models are placed and loaded by the rig manager
(llm-gateway `rigctl load-model --use benchmark`, which puts the rig in bench mode): a block names
no cards (the manager picks the fewest that hold the model at its registry profile's context) or pins
them (`gpus: [0, 1, 2, 3]`, `placement: brain` or `gpu_1_3`). See
plans/shoulders/benchmarking/docker/CAMPAIGN_RUNNER_REFACTOR.md.

Usage:
    python3 /mnt/shared/scripts/benchmarks/run_campaign.py \\
        /mnt/shared/plans/shoulders/benchmarking/campaigns/my_campaign.json \\
        [--run-id ID] [--dry-run] [--on-failure continue|stop] [--verbose]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TICK_INTERVAL = 3  # seconds between scheduler ticks
SHARED_ROOT = Path(os.environ.get("CAMPAIGN_SHARED_ROOT", "/mnt/shared"))
TUNING_PROFILES_PATH = SHARED_ROOT / "plans" / "shoulders" / "benchmarking" / "model_tuning_profiles.json"
SUITE_CONTRACTS_PATH = SHARED_ROOT / "plans" / "shoulders" / "benchmarking" / "docker" / "suite_contracts.json"
MODELS_CATALOG_PATH = SHARED_ROOT / "agents" / "models.catalog.json"
HISTORY_ROOT = SHARED_ROOT / "logs" / "benchmarks" / "campaigns" / "history"
BENCHMARKING_SCRIPTS = SHARED_ROOT / "plans" / "shoulders" / "benchmarking"
# The rig manager (llm-gateway docs/MANAGER.md) on the Pi, reached with rigctl over ssh like the lanes.
MANAGER_HOST = os.environ.get("HARNESS_HOST", "pi")
RIGCTL = os.environ.get("HARNESS_RIGCTL", "~/llm-gateway/tools/rigctl.py")
# How long a load may wait for its cards to free (the manager collects them meanwhile); a block's
# `wait_s` overrides it, -1 waits forever.
LOAD_WAIT_S = 7200
RIG_CONTAINER_PREFIX = "rig-remote-"  # the rig names a manager load's container after its profile
BENCHMARK_CACHE_ROOT = SHARED_ROOT / "cache" / "benchmarks"
SUITE_CONTRACT_LABEL = "daedalmap.benchmark.contract"
SUITE_CONTRACT_VERSION = "campaign-v1"
SUITE_SPECS: Dict[str, Dict[str, Any]] = {}
WORKER_SUITES: Set[str] = set()
STANDALONE_SUITES: Set[str] = set()
SUPPORTED_SUITES: Set[str] = set()

# Block states
PENDING = "pending"
WAITING_DEPS = "waiting_deps"
WAITING_GPU = "waiting_gpu"
LOADING = "loading"
RUNNING_SUITE = "running_suite"
COMPLETED = "completed"
FAILED = "failed"


# ---------------------------------------------------------------------------
# Runtimes: what the rig manager loaded for a runtime group
# ---------------------------------------------------------------------------

@dataclass
class Runtime:
    """A model the rig manager loaded: its profile id, the port it serves on, its cards and context."""
    profile: str
    port: int
    gpu_ids: List[int]
    ctx: Optional[int] = None

    @property
    def name(self) -> str:
        """The cards as the old slot names (brain, gpu_2, gpu_1_3): status and hardware ids read the same."""
        if not self.gpu_ids:
            return "manager-picked"  # a dry-run stand-in before the manager has chosen cards
        return "brain" if self.gpu_ids == [0] else "gpu_" + "_".join(str(g) for g in self.gpu_ids)

    @property
    def container(self) -> str:
        return RIG_CONTAINER_PREFIX + self.profile


def block_gpus(block: Dict[str, Any]) -> Optional[List[int]]:
    """The cards a block pins, or None when the manager picks them.

    `gpus: [0, 1]` pins those cards; `placement: brain` is GPU 0 and `placement: gpu_1_3` GPUs 1 and 3 (the
    old slot names). `single` and `split` are gone: one card or several is the manager's answer.
    """
    if block.get("gpus") is not None:
        return sorted({int(g) for g in block["gpus"]})
    placement = block.get("placement")
    if placement in (None, ""):
        return None
    if placement == "brain":
        return [0]
    match = re.fullmatch(r"gpu((?:_\d+)+)", str(placement))
    if match:
        return sorted({int(g) for g in match.group(1).strip("_").split("_")})
    if placement in ("single", "split"):
        raise ValueError(f"placement '{placement}' is gone: omit placement and the rig manager picks the cards")
    raise ValueError(f"invalid placement '{placement}': omit it, or use gpus: [..], brain or gpu_<ids>")


def rigctl_cmd(*args: str) -> List[str]:
    """rigctl on the Pi over ssh; ssh re-splits the remote line, so every argument is quoted for it."""
    return ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", MANAGER_HOST, "python3", RIGCTL,
            *[shlex.quote(str(a)) for a in args]]


def load_cmd(block: "BlockState", owner: str) -> List[str]:
    """`rigctl load-model` for a block: use benchmark (bench mode, the registry profile's context), the
    block's cards if it pins them, and only the settings the block overrides."""
    cfg = block.config
    wait = cfg.get("wait_s", LOAD_WAIT_S)
    args = ["load-model", cfg["model"], "--use", "benchmark", "--json", "--owner", owner,
            "--wait", "forever" if int(wait) < 0 else str(int(wait))]
    if block.gpus:
        args += ["--gpus", ",".join(str(g) for g in block.gpus)]
    if cfg.get("ctx_size"):
        args += ["--ctx", str(int(cfg["ctx_size"]))]
    if cfg.get("profile"):
        args += ["--profile", str(cfg["profile"])]
    if cfg.get("runtime_args"):
        args += ["--args", shlex.join(str(a) for a in cfg["runtime_args"])]
    return rigctl_cmd(*args)


def start_load(block: "BlockState", owner: str) -> subprocess.Popen:
    """Ask the manager to load a block's model; the answer comes when the model serves (or it fails)."""
    cmd = load_cmd(block, owner)
    log(f"  Load: {' '.join(cmd)}", verbose_only=True)
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def finish_load(proc: subprocess.Popen) -> tuple[Runtime, str]:
    """(runtime, GGUF file name the manager loaded) from a finished load. RuntimeError with the manager's
    reason when it refused or failed (won't fit, the rig busy in another mode, cards never freed)."""
    out, err = proc.communicate()
    try:
        answer = json.loads(out or "{}")
    except ValueError:
        answer = {}
    if proc.returncode != 0 or answer.get("state") != "ready":
        message = (answer.get("error") or {}).get("message") or (err or out or "").strip()[-500:]
        raise RuntimeError(f"rig manager did not load it (exit {proc.returncode}): {message}")
    gguf = Path(str((answer.get("estimate") or {}).get("gguf_path", ""))).name
    return Runtime(profile=answer["profile"], port=int(answer["port"]), gpu_ids=[int(g) for g in answer["gpus"]],
                   ctx=answer.get("ctx")), gguf


def unload_runtime(runtime: Optional[Runtime]) -> None:
    """Unload a runtime through the manager (it releases the lease; the rig goes idle with the last one)."""
    if runtime is None or runtime.port == 0:  # port 0: a dry-run stand-in
        return
    log(f"  Unloading {runtime.profile}")
    result = subprocess.run(rigctl_cmd("unload", runtime.profile), capture_output=True, text=True,
                            timeout=900, check=False)
    if result.returncode != 0:
        log(f"  WARNING: unload of {runtime.profile} failed: {(result.stderr or result.stdout).strip()[-300:]}")


def cancel_owner(owner: str, reason: str) -> None:
    """Everything the run still owns in the manager: waits, loads, leases, claims (MANAGER.md)."""
    result = subprocess.run(rigctl_cmd("cancel-owner", owner, "--reason", reason), capture_output=True,
                            text=True, timeout=900, check=False)
    if result.returncode != 0:
        log(f"  WARNING: cancel-owner {owner} failed: {(result.stderr or result.stdout).strip()[-300:]}")


# ---------------------------------------------------------------------------
# Block State
# ---------------------------------------------------------------------------

@dataclass
class BlockState:
    block_id: str
    config: Dict[str, Any]
    status: str = PENDING
    runtime: Optional[Runtime] = None
    suite_process: Optional[subprocess.Popen] = None
    suite_log_path: Optional[Path] = None
    started_at: Optional[str] = None
    load_started_at: Optional[str] = None
    suite_started_at: Optional[str] = None
    ended_at: Optional[str] = None
    exit_code: Optional[int] = None
    error: Optional[str] = None
    identity: Optional[Dict[str, Any]] = None  # verified runtime identity evidence

    @property
    def gpus(self) -> Optional[List[int]]:
        """Pinned cards, or None when the manager picks them."""
        return block_gpus(self.config)

    @property
    def placement(self) -> str:
        return ",".join(str(g) for g in self.gpus) if self.gpus else "manager"

    @property
    def suite(self) -> str:
        return self.config["suite"]

    @property
    def model(self) -> str:
        return self.config["model"]

    @property
    def depends_on(self) -> List[str]:
        return self.config.get("depends_on", [])


@dataclass
class CampaignState:
    name: str
    run_id: str
    manifest_path: str
    blocks: Dict[str, BlockState]
    started_at: str
    status: str = "running"
    ended_at: Optional[str] = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def log(msg: str, verbose_only: bool = False) -> None:
    """Print a timestamped log line."""
    if verbose_only and not _VERBOSE:
        return
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def write_json_atomic(path: Path, data: Any) -> None:
    """Write JSON with atomic rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)


def docker_stop(container_name: str) -> None:
    """Stop and remove a suite container by name (models are the manager's: unload_runtime)."""
    subprocess.run(
        ["docker", "stop", container_name],
        capture_output=True, text=True, check=False,
    )
    # --rm flag on the container handles removal, but try anyway
    subprocess.run(
        ["docker", "rm", "-f", container_name],
        capture_output=True, text=True, check=False,
    )


def load_suite_contracts() -> None:
    """Load the single source of truth for campaign-compatible suites."""
    global SUITE_SPECS, WORKER_SUITES, STANDALONE_SUITES, SUPPORTED_SUITES
    if not SUITE_CONTRACTS_PATH.is_file():
        raise SystemExit(f"ERROR: suite contract registry missing: {SUITE_CONTRACTS_PATH}")
    data = json.loads(SUITE_CONTRACTS_PATH.read_text(encoding="utf-8"))
    if data.get("contract_version") != SUITE_CONTRACT_VERSION:
        raise SystemExit(
            f"ERROR: suite contract version must be {SUITE_CONTRACT_VERSION}: "
            f"{SUITE_CONTRACTS_PATH}"
        )
    specs = data.get("suites")
    if not isinstance(specs, dict) or not specs:
        raise SystemExit(f"ERROR: suite contract registry has no suites: {SUITE_CONTRACTS_PATH}")
    SUITE_SPECS = specs
    WORKER_SUITES = {
        name for name, spec in specs.items() if spec.get("runtime_mode") == "external"
    }
    STANDALONE_SUITES = {
        name for name, spec in specs.items() if spec.get("runtime_mode") == "standalone"
    }
    SUPPORTED_SUITES = set(specs)


def suite_results_root(suite: str) -> Path:
    return SHARED_ROOT / "logs" / "benchmarks" / suite / "history"


def preflight_suite_images(suites: Set[str]) -> None:
    """Require images built against the campaign runner CLI contract."""
    errors = []
    for suite in sorted(suites):
        result = subprocess.run(
            [
                "docker", "image", "inspect", suite,
                "--format", f"{{{{ index .Config.Labels \"{SUITE_CONTRACT_LABEL}\" }}}}",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        actual = result.stdout.strip() if result.returncode == 0 else "missing"
        if actual != SUITE_CONTRACT_VERSION:
            errors.append(f"{suite}={actual or 'unlabeled'}")
    if errors:
        raise SystemExit(
            "ERROR: stale or incompatible benchmark suite images: "
            + ", ".join(errors)
            + ". Rebuild them from the benchmarking/docker directory."
        )


def preflight_output_roots(suites: Set[str]) -> None:
    """Verify suite result roots are writable by the campaign container UID."""
    errors = []
    roots = [suite_results_root(suite) for suite in sorted(suites)]
    roots.append(BENCHMARK_CACHE_ROOT)
    for root in roots:
        try:
            root.mkdir(parents=True, exist_ok=True)
            probe = root / f".campaign-write-probe-{os.getpid()}"
            probe.write_text("ok\n", encoding="utf-8")
            probe.unlink()
        except OSError as exc:
            errors.append(f"{root}: {exc}")
    if errors:
        raise SystemExit(
            "ERROR: benchmark result roots are not writable by the campaign user: "
            + "; ".join(errors)
        )


def served_model_id(port: int, timeout: int = 5) -> Optional[str]:
    """GGUF file name the llama-server on this port reports, or None if nothing answers."""
    try:
        import urllib.request
        url = f"http://127.0.0.1:{port}/v1/models"
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8", errors="replace"))
            models = data.get("data", [])
            if not models:
                return None
            return Path(str(models[0].get("id", ""))).name or "?"
    except Exception:
        return None


def container_info(container_name: Optional[str]) -> Optional[Dict[str, str]]:
    """Running state, id and image of a container, or None if it does not exist."""
    if not container_name:
        return None
    result = subprocess.run(
        ["docker", "inspect", "-f", "{{.State.Running}}|{{.Id}}|{{.Config.Image}}", container_name],
        capture_output=True, text=True, check=False,
    )
    if result.returncode != 0:
        return None
    running, container_id, image = (result.stdout.strip().split("|") + ["", "", ""])[:3]
    return {"running": running, "id": container_id[:12], "image": image}


def runtime_identity_error(block: "BlockState", runtime: Runtime, observed: Optional[str]) -> Optional[str]:
    """Why the served runtime is not the one this block requested, or None when it is.

    Health alone is not readiness: a stale server on the same port answers too
    (campaign 20260820_modern_validation_l50_retry3). On success the evidence is
    stored on the block for status.json and the checkpoint.
    """
    expected = Path(str(block.config["gguf"])).name
    if observed is None:
        return f"no runtime answers on port {runtime.port}"
    if observed != expected:
        return f"port {runtime.port} serves {observed}, expected {expected}"
    info = container_info(runtime.container)
    if not info or info["running"] != "true":
        return (f"port {runtime.port} serves {observed} but owned runtime "
                f"{runtime.container} is not running")
    block.identity = {
        "requested": expected, "observed": observed, "port": runtime.port, "profile": runtime.profile,
        "gpus": runtime.gpu_ids, "ctx": runtime.ctx, "container": runtime.container,
        "container_id": info["id"], "image": info["image"], "verified_at": now_iso(),
    }
    return None


# ---------------------------------------------------------------------------
# Config Lookups
# ---------------------------------------------------------------------------

_tuning_cache: Optional[Dict] = None
_catalog_cache: Optional[Dict] = None


def _load_tuning_profiles() -> Dict:
    global _tuning_cache
    if _tuning_cache is None:
        if TUNING_PROFILES_PATH.exists():
            _tuning_cache = json.loads(TUNING_PROFILES_PATH.read_text(encoding="utf-8"))
        else:
            _tuning_cache = {}
    return _tuning_cache


def _load_catalog() -> Dict:
    global _catalog_cache
    if _catalog_cache is None:
        if MODELS_CATALOG_PATH.exists():
            _catalog_cache = json.loads(MODELS_CATALOG_PATH.read_text(encoding="utf-8"))
        else:
            _catalog_cache = {}
    return _catalog_cache


def _norm(v: str) -> str:
    s = v.strip().lower()
    if s.endswith(".gguf"):
        s = s[:-5]
    return re.sub(r"[^a-z0-9]+", "", s)


def _lookup_tuning(model_id: str) -> Optional[Dict]:
    """Resolve a model ID, including aliases, in tuning profiles."""
    profiles = _load_tuning_profiles()
    models = profiles.get("models", {})
    profiles_module_path = BENCHMARKING_SCRIPTS / "scripts" / "active"
    if str(profiles_module_path) not in sys.path:
        sys.path.insert(0, str(profiles_module_path))
    from model_profiles import resolve_profile

    resolved = resolve_profile(models, model_id)
    return resolved[1] if resolved else None


def registry_gguf(model_id: str, profile: Optional[str] = None) -> str:
    """The archive GGUF the model library names for a model (the manager loads the same file)."""
    library_dir = BENCHMARKING_SCRIPTS / "scripts" / "active"
    if str(library_dir) not in sys.path:
        sys.path.insert(0, str(library_dir))
    import model_library

    registry = model_library.load_registry(BENCHMARKING_SCRIPTS / "model_registry.json")
    return str(model_library.resolve(model_id, profile or "default", registry)["gguf_path"])


def resolve_block_config(block: Dict, defaults: Dict) -> Dict:
    """Merge block config with campaign defaults. Runtime settings (context, batch, image, registry extra
    args) are the manager's from the registry profile; a block carries only what it overrides."""
    # gguf: the registry's. A manifest gguf is kept only as a cross-check (validate_resolved_blocks).
    try:
        registry_path = registry_gguf(block["model"], block.get("profile"))
    except KeyError as exc:
        raise SystemExit(f"ERROR: block '{block['id']}': {exc.args[0] if exc.args else exc}")
    if block.get("gguf") and Path(str(block["gguf"])).name != Path(registry_path).name:
        block["_gguf_mismatch"] = f"manifest gguf {block['gguf']} is not the registry's {registry_path}"
    block["gguf"] = registry_path
    for gone in ("runtime_image", "batch_size", "load_timeout_s"):
        if block.pop(gone, None) is not None:
            log(f"  {block['id']}: manifest {gone} ignored (the manager uses the registry profile)", verbose_only=True)

    # runtime_args: only the block's own; the manager adds the registry profile's extra_args first
    block["runtime_args"] = list(block.get("runtime_args", []))

    # depends_on: default empty
    if "depends_on" not in block:
        block["depends_on"] = []

    # placement: pinned cards (gpus, brain, gpu_<ids>) or none, so the manager picks them
    try:
        block_gpus(block)
    except ValueError as exc:
        raise SystemExit(f"ERROR: block '{block['id']}': {exc}")

    # suite_args: default empty
    if "suite_args" not in block:
        block["suite_args"] = []

    # The runner owns run identity and class. Older manifests placed run-class
    # in suite_args; normalize it once so every suite receives one value.
    suite_args = list(block["suite_args"])
    normalized_args: List[str] = []
    index = 0
    while index < len(suite_args):
        arg = str(suite_args[index])
        if arg == "--run-class":
            if index + 1 >= len(suite_args):
                raise SystemExit(f"ERROR: block '{block['id']}' has --run-class without a value")
            block.setdefault("run_class", str(suite_args[index + 1]))
            index += 2
            continue
        normalized_args.append(arg)
        index += 1
    block["suite_args"] = normalized_args
    block.setdefault("run_class", defaults.get("run_class", "provisional"))

    return block


def resolve_limit(block: Dict) -> List[str]:
    """Turn the limit convenience field into --limit N args.

    Returns list of args to append to suite_args.
    Explicit --limit in suite_args wins over the limit field.
    """
    suite_args = block.get("suite_args", [])
    if "--limit" in suite_args:
        return []  # explicit wins

    limit = block.get("limit")
    if limit is not None:
        return ["--limit", str(limit)]

    return []


def validate_resolved_blocks(blocks: Dict[str, BlockState]) -> None:
    """Fail before model loading when a block violates its suite contract."""
    errors: List[str] = []
    run_classes = {"smoke", "provisional", "validated", "full"}
    runner_owned = {
        "--model", "--runtime-base", "--run-name", "--results-dir",
        "--scripts-dir", "--tuning-profiles", "--hardware-id", "--gguf",
    }
    for bid, block in blocks.items():
        cfg = block.config
        suite = cfg["suite"]
        spec = SUITE_SPECS[suite]
        gguf = Path(str(cfg.get("gguf", "")))
        if not gguf.is_file():
            errors.append(f"{bid}: GGUF not found: {gguf}")
        if cfg.get("_gguf_mismatch"):
            errors.append(f"{bid}: {cfg['_gguf_mismatch']}")
        run_class = str(cfg.get("run_class", ""))
        if run_class not in run_classes:
            errors.append(f"{bid}: invalid run_class '{run_class}'")
        suite_args = [str(value) for value in cfg.get("suite_args", [])]
        conflicts = sorted(runner_owned.intersection(suite_args))
        if conflicts:
            errors.append(f"{bid}: suite_args contains runner-managed flags: {', '.join(conflicts)}")
        if cfg.get("limit") is not None and not spec.get("limit_supported"):
            errors.append(f"{bid}: {suite} does not support bounded --limit runs")
        profile = _lookup_tuning(cfg["model"])
        policy = spec.get("profile_policy")
        if policy in {"required", "task_prompt_plus_profile"} and profile is None:
            errors.append(f"{bid}: no unique model profile for {cfg['model']}")
        if policy == "required" and profile is not None:
            if not str(profile.get("system_prompt", "")).strip():
                errors.append(f"{bid}: resolved model profile has no system_prompt")
        if spec.get("env_file_supported") and cfg.get("env_file"):
            if not Path(str(cfg["env_file"])).is_file():
                errors.append(f"{bid}: env_file not found: {cfg['env_file']}")

    groups: Dict[str, List[BlockState]] = {}
    for block in blocks.values():
        group = str(block.config.get("runtime_group", "")).strip()
        if group:
            groups.setdefault(group, []).append(block)
    runtime_keys = ("model", "gguf", "profile", "ctx_size", "runtime_args", "wait_s")

    def runtime_settings(block: BlockState) -> Dict[str, Any]:
        return {**{key: block.config.get(key) for key in runtime_keys}, "gpus": block.gpus}

    for group, members in groups.items():
        if len(members) < 2:
            errors.append(f"runtime_group '{group}' must contain at least two blocks")
            continue
        expected = runtime_settings(members[0])
        for member in members[1:]:
            actual = runtime_settings(member)
            if actual != expected:
                errors.append(
                    f"{member.block_id}: runtime_group '{group}' changes runtime settings"
                )
    if errors:
        raise SystemExit("ERROR: campaign preflight failed:\n  - " + "\n  - ".join(errors))


# ---------------------------------------------------------------------------
# Suite Layer (wraps docker run bench-*)
# ---------------------------------------------------------------------------

def build_suite_cmd(block: BlockState, runtime: Runtime) -> List[str]:
    """Build the docker run command for a benchmark suite against the runtime the manager loaded."""
    cfg = block.config
    suite = cfg["suite"]
    spec = SUITE_SPECS[suite]
    container_name = f"bench-{suite.replace('bench-', '')}-campaign-{block.block_id}"
    model_id = cfg["model"]
    port = runtime.port
    run_id = cfg.get("_run_id", "unknown")
    run_name = f"campaign_{block.block_id}_{run_id}"
    suite_log_root = suite_results_root(suite)

    # Resolve limit
    extra_limit_args = resolve_limit(cfg)
    suite_args = list(cfg.get("suite_args", [])) + extra_limit_args
    run_class = str(cfg["run_class"])
    gpu_spec = ",".join(str(g) for g in runtime.gpu_ids)
    hardware_id = str(
        cfg.get("hardware_id")
        or f"rig-{runtime.name}-gpu{'-'.join(str(g) for g in runtime.gpu_ids)}"
    )

    if suite in WORKER_SUITES:
        cmd = [
            "docker", "run", "--rm",
            "--name", container_name,
            "--network", "host",
            "--user", f"{os.getuid()}:{os.getgid()}",
            "-e", "BENCHMARK_DISABLE_AUTO_RESERVE=1",
            "-e", "HOME=/tmp/benchmark-home",
            "-e", f"HF_HOME={BENCHMARK_CACHE_ROOT / 'huggingface'}",
            "-e", f"HF_DATASETS_CACHE={BENCHMARK_CACHE_ROOT / 'huggingface' / 'datasets'}",
            "-e", f"XDG_CACHE_HOME={BENCHMARK_CACHE_ROOT / 'xdg'}",
            "-e", f"NLTK_DATA={BENCHMARK_CACHE_ROOT / 'nltk_data'}",
            "-v", f"{SHARED_ROOT}:{SHARED_ROOT}",
            "-v", f"{suite_log_root}:/results",
            "-v", f"{BENCHMARKING_SCRIPTS}:/benchmark-scripts:ro",
            # One EvalPlus scoring at a time on the rig (bench-code run.sh; peaks of 4.7 GB RAM each).
            "-v", "/run/lock:/host-lock",
            "-e", "BENCH_EVAL_LOCK=/host-lock/bench-evalplus.lock",
        ]
        if spec.get("container_gpu_access"):
            docker_gpu_spec = f'"device={gpu_spec}"' if len(runtime.gpu_ids) > 1 else f"device={gpu_spec}"
            cmd.extend(["--gpus", docker_gpu_spec])
        if spec.get("env_file_supported") and cfg.get("env_file"):
            cmd.extend(["--env-file", str(cfg["env_file"])])
        cmd.extend([
            suite,
            "--model", model_id,
            "--runtime-base", f"http://localhost:{port}",
            "--run-name", run_name,
            "--run-class", run_class,
            "--results-dir", "/results",
        ])
        if spec.get("profile_policy") in {"required", "task_prompt_plus_profile"}:
            cmd.extend([
                "--scripts-dir", "/benchmark-scripts",
                "--tuning-profiles", "/benchmark-scripts/model_tuning_profiles.json",
            ])
        if spec.get("hardware_id_required"):
            cmd.extend(["--hardware-id", hardware_id])
        if suite == "bench-runtime":
            # NVIDIA_VISIBLE_DEVICES is renumbered from zero inside the suite
            # container. Preserve host IDs in hardware_id, but query telemetry
            # using the container-local namespace.
            local_gpu_spec = ",".join(str(i) for i in range(len(runtime.gpu_ids)))
            cmd.extend(["--gpu-ids", local_gpu_spec])
        cmd.extend(suite_args)

    else:
        raise ValueError(f"Unsupported suite: {suite}")

    return cmd


def start_suite(block: BlockState, runtime: Runtime) -> subprocess.Popen:
    """Launch the suite container as a subprocess."""
    cmd = build_suite_cmd(block, runtime)
    container_name = f"bench-{block.suite.replace('bench-', '')}-campaign-{block.block_id}"

    # Stop any leftover suite container
    docker_stop(container_name)

    # Set up log capture
    log_dir = HISTORY_ROOT / block.config.get("_campaign_name", "unknown") / block.config.get("_run_id", "unknown") / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{block.block_id}.log"

    log(f"  Starting suite: {container_name}")
    log(f"  Command: {' '.join(cmd)}", verbose_only=True)
    log(f"  Log: {log_path}")

    log_file = open(log_path, "w", encoding="utf-8")
    proc = subprocess.Popen(
        cmd,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        text=True,
    )

    block.suite_log_path = log_path
    # Store log file handle for cleanup
    block.config["_log_file"] = log_file

    return proc


# ---------------------------------------------------------------------------
# Status & Checkpoint
# ---------------------------------------------------------------------------

def build_status_dict(state: CampaignState) -> Dict[str, Any]:
    """Build the status.json representation."""
    blocks_status = {}
    for bid, bs in state.blocks.items():
        entry: Dict[str, Any] = {"status": bs.status}
        if bs.runtime:
            entry["gpu_slot"] = bs.runtime.name
            entry["port"] = bs.runtime.port
            entry["profile"] = bs.runtime.profile
        entry["suite"] = bs.suite
        if bs.started_at:
            entry["started_at"] = bs.started_at
        if bs.ended_at:
            entry["ended_at"] = bs.ended_at
        if bs.exit_code is not None:
            entry["exit_code"] = bs.exit_code
        if bs.error:
            entry["error"] = bs.error
        if bs.identity:
            entry["runtime_identity"] = bs.identity
        if bs.suite_log_path:
            entry["log_path"] = str(bs.suite_log_path)

        # Provide waiting_for info
        if bs.status == WAITING_GPU:
            loading = [obid for obid, obs in state.blocks.items() if obs.status == LOADING]
            if loading:
                entry["waiting_for"] = f"manager load in flight ({', '.join(loading)})"
        if bs.status == LOADING:
            entry["waiting_for"] = "the rig manager: placing and loading (it waits for busy cards)"
        if bs.status == WAITING_DEPS:
            pending_deps = [
                d for d in bs.depends_on
                if d in state.blocks and state.blocks[d].status not in (COMPLETED,)
            ]
            if pending_deps:
                entry["waiting_for"] = f"depends_on: {', '.join(pending_deps)}"

        blocks_status[bid] = entry

    return {
        "campaign": state.name,
        "run_id": state.run_id,
        "status": state.status,
        "started_at": state.started_at,
        "blocks": blocks_status,
        "last_updated": now_iso(),
    }


def write_status(state: CampaignState) -> None:
    """Write status.json atomically."""
    path = HISTORY_ROOT / state.name / state.run_id / "status.json"
    write_json_atomic(path, build_status_dict(state))


def write_checkpoint(state: CampaignState) -> None:
    """Write campaign_state.json for resume."""
    path = HISTORY_ROOT / state.name / state.run_id / "campaign_state.json"

    blocks_data = {}
    for bid, bs in state.blocks.items():
        blocks_data[bid] = {
            "status": bs.status,
            "slot": bs.runtime.name if bs.runtime else None,
            "profile": bs.runtime.profile if bs.runtime else None,
            "started_at": bs.started_at,
            "ended_at": bs.ended_at,
            "exit_code": bs.exit_code,
            "error": bs.error,
            "runtime_identity": bs.identity,
        }

    data = {
        "campaign": state.name,
        "run_id": state.run_id,
        "manifest_path": state.manifest_path,
        "status": state.status,
        "started_at": state.started_at,
        "ended_at": state.ended_at,
        "blocks": blocks_data,
        "saved_at": now_iso(),
    }
    write_json_atomic(path, data)


def load_checkpoint(state: CampaignState) -> bool:
    """Load checkpoint and mark completed blocks. Returns True if checkpoint found."""
    path = HISTORY_ROOT / state.name / state.run_id / "campaign_state.json"
    if not path.exists():
        return False

    data = json.loads(path.read_text(encoding="utf-8"))
    saved_blocks = data.get("blocks", {})

    for bid, saved in saved_blocks.items():
        if bid not in state.blocks:
            continue
        saved_status = saved.get("status", "")
        if saved_status == COMPLETED:
            bs = state.blocks[bid]
            bs.status = COMPLETED
            bs.started_at = saved.get("started_at")
            bs.ended_at = saved.get("ended_at")
            bs.exit_code = saved.get("exit_code")
            log(f"  Checkpoint: block {bid} already completed, skipping")
        elif saved_status in (LOADING, RUNNING_SUITE):
            # Was mid-flight — restart from beginning
            log(f"  Checkpoint: block {bid} was {saved_status}, will restart")
            # Unload the orphan runtime the interrupted run left (the manager knows it by profile).
            orphan = saved.get("profile")
            if orphan:
                unload_runtime(Runtime(profile=orphan, port=-1, gpu_ids=[]))

    return True


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------

class Scheduler:
    """Runs the blocks: one manager load request at a time (the rig loads one model at a time anyway, and
    two multi-card loads collecting cards at once could each hold cards the other needs), any number of
    loaded runtime groups side by side, suites against the ports the manager returned."""

    def __init__(
        self,
        state: CampaignState,
        on_failure: str = "continue",
        dry_run: bool = False,
        max_active_models: Optional[int] = None,
    ):
        self.state = state
        self.on_failure = on_failure
        self.dry_run = dry_run
        self.max_active_models = max_active_models
        self.owner = re.sub(r"[^A-Za-z0-9._-]+", "-", f"campaign-{state.run_id}")[:64]
        self.loading_block: Optional[str] = None
        self.load_proc: Optional[subprocess.Popen] = None
        self._shutdown = False

    def _runtime_group_members(self, group: str) -> List[BlockState]:
        return [
            block for block in self.state.blocks.values()
            if str(block.config.get("runtime_group", "")).strip() == group
        ]

    def _runtime_group_predecessor(self, block: BlockState) -> Optional[BlockState]:
        group = str(block.config.get("runtime_group", "")).strip()
        if not group:
            return None
        members = self._runtime_group_members(group)
        index = members.index(block)
        return members[index - 1] if index > 0 else None

    def _runtime_group_successor(self, block: BlockState) -> Optional[BlockState]:
        group = str(block.config.get("runtime_group", "")).strip()
        if not group:
            return None
        members = self._runtime_group_members(group)
        index = members.index(block)
        return next(
            (candidate for candidate in members[index + 1:] if candidate.status != COMPLETED),
            None,
        )

    def _active_model_groups(self) -> Set[str]:
        groups = set()
        for block in self.state.blocks.values():
            if block.status not in (LOADING, RUNNING_SUITE):
                continue
            group = str(block.config.get("runtime_group", "")).strip()
            groups.add(group or block.block_id)
        return groups

    def _end_runtime(self, block: BlockState) -> None:
        """Unload the block's runtime (the end of its group, or a failure)."""
        if not self.dry_run:
            unload_runtime(block.runtime)
        block.runtime = None

    def handoff_runtime(self, block: BlockState) -> bool:
        """Start the next suite in a runtime group without reloading its model."""
        successor = self._runtime_group_successor(block)
        runtime = block.runtime
        if successor is None or runtime is None:
            return False
        if successor.status not in (PENDING, WAITING_DEPS, WAITING_GPU):
            return False
        if any(
            self.state.blocks[dep].status != COMPLETED
            for dep in successor.depends_on
            if dep in self.state.blocks
        ):
            return False
        if not self.dry_run:
            # The successor inherits the runtime, so it must pass the same identity check.
            problem = runtime_identity_error(successor, runtime, served_model_id(runtime.port))
            if problem:
                log(f"Block {block.block_id}: not reusing runtime ({problem}); successor will reload it")
                self._end_runtime(block)
                return False

        successor.runtime = runtime
        successor.started_at = now_iso()
        successor.suite_started_at = now_iso()
        successor.status = RUNNING_SUITE
        block.runtime = None
        group = str(successor.config["runtime_group"])
        log(
            f"Block {successor.block_id} -> {runtime.name} "
            f"[reusing runtime group {group}]"
        )
        if self.dry_run:
            log(
                f"  Suite command: {' '.join(build_suite_cmd(successor, runtime))}",
                verbose_only=True,
            )
            return True
        try:
            successor.suite_process = start_suite(successor, runtime)
            return True
        except Exception as exc:
            successor.status = FAILED
            successor.error = f"suite start failed: {exc}"
            successor.ended_at = now_iso()
            self._end_runtime(successor)
            log(f"Block {successor.block_id}: FAILED to start suite: {exc}")
            return False

    def _all_done(self) -> bool:
        return all(
            bs.status in (COMPLETED, FAILED)
            for bs in self.state.blocks.values()
        )

    def _any_failed(self) -> bool:
        return any(
            bs.status == FAILED
            for bs in self.state.blocks.values()
        )

    def advance_deps(self) -> None:
        """Move blocks from pending -> waiting_deps -> waiting_gpu as deps resolve."""
        for bid, bs in self.state.blocks.items():
            if bs.status == PENDING:
                bs.status = WAITING_DEPS

            if bs.status == WAITING_DEPS:
                predecessor = self._runtime_group_predecessor(bs)
                if predecessor and predecessor.status not in (COMPLETED, FAILED):
                    continue
                deps_met = all(
                    self.state.blocks[d].status == COMPLETED
                    for d in bs.depends_on
                    if d in self.state.blocks
                )
                # Also check for failed deps — fail this block too
                deps_failed = any(
                    self.state.blocks[d].status == FAILED
                    for d in bs.depends_on
                    if d in self.state.blocks
                )
                if deps_failed:
                    bs.status = FAILED
                    bs.error = "dependency failed"
                    bs.ended_at = now_iso()
                    log(f"Block {bid}: FAILED (dependency failed)")
                elif deps_met:
                    bs.status = WAITING_GPU

    def try_start_load(self) -> None:
        """Ask the manager to load the next ready block's model (one request in flight at a time)."""
        if self.loading_block:
            return

        for bid, bs in self.state.blocks.items():
            if bs.status != WAITING_GPU:
                continue

            if self.max_active_models is not None:
                group = str(bs.config.get("runtime_group", "")).strip() or bs.block_id
                active_groups = self._active_model_groups()
                if group not in active_groups and len(active_groups) >= self.max_active_models:
                    continue

            bs.status = LOADING
            bs.started_at = now_iso()
            bs.load_started_at = now_iso()
            self.loading_block = bid
            log(f"Block {bid}: asking the rig manager to load {bs.model} (cards: {bs.placement})")

            if self.dry_run:
                log(f"  Load command: {' '.join(load_cmd(bs, self.owner))}", verbose_only=True)
                return
            try:
                self.load_proc = start_load(bs, self.owner)
            except Exception as exc:
                bs.status = FAILED
                bs.error = f"load request failed: {exc}"
                bs.ended_at = now_iso()
                self.loading_block = None
                log(f"Block {bid}: FAILED to ask for a load: {exc}")
            return  # only one load request per tick

    def check_load_complete(self) -> None:
        """When the manager answers the load in flight: verify the runtime and start the block's suite."""
        if not self.loading_block:
            return

        bs = self.state.blocks[self.loading_block]
        if bs.status != LOADING:
            # Something else changed the status (e.g., shutdown)
            self.loading_block = None
            self.load_proc = None
            return

        if self.dry_run:
            # In dry-run, instantly "complete" the load on a stand-in runtime
            bs.runtime = Runtime(profile=f"dry-run-{bs.block_id}", port=0, gpu_ids=bs.gpus or [])
            bs.status = RUNNING_SUITE
            bs.suite_started_at = now_iso()
            self.loading_block = None
            log(f"Block {bs.block_id}: [dry-run] model loaded, starting suite")
            log(f"  Suite command: {' '.join(build_suite_cmd(bs, bs.runtime))}", verbose_only=True)
            return

        if self.load_proc is None or self.load_proc.poll() is None:
            return  # the manager is still placing, waiting for cards, or loading
        proc, self.load_proc, self.loading_block = self.load_proc, None, None
        try:
            runtime, served_gguf = finish_load(proc)
        except RuntimeError as exc:
            bs.status = FAILED
            bs.error = str(exc)
            bs.ended_at = now_iso()
            log(f"Block {bs.block_id}: FAILED, {exc}")
            return
        bs.runtime = runtime

        observed = served_model_id(runtime.port)
        mismatch = runtime_identity_error(bs, runtime, observed)
        if not mismatch and served_gguf and served_gguf != Path(str(bs.config["gguf"])).name:
            mismatch = f"the manager loaded {served_gguf}, expected {Path(str(bs.config['gguf'])).name}"
        if mismatch:
            log(f"Block {bs.block_id}: IDENTITY MISMATCH, suite not started: {mismatch}")
            bs.status = FAILED
            bs.error = f"runtime identity mismatch: {mismatch}"
            bs.ended_at = now_iso()
            self._end_runtime(bs)
            return

        log(f"Block {bs.block_id}: model ready as {runtime.profile} on port {runtime.port} "
            f"(GPU {runtime.gpu_ids}, ctx {runtime.ctx}), identity verified {observed} "
            f"in {bs.identity['container_id']}")
        bs.status = RUNNING_SUITE
        bs.suite_started_at = now_iso()
        try:
            bs.suite_process = start_suite(bs, runtime)
        except Exception as e:
            bs.status = FAILED
            bs.error = f"suite start failed: {e}"
            bs.ended_at = now_iso()
            self._end_runtime(bs)
            log(f"Block {bs.block_id}: FAILED to start suite: {e}")

    def reap_finished(self) -> None:
        """Poll running suite processes for completion."""
        for bid, bs in self.state.blocks.items():
            if bs.status != RUNNING_SUITE:
                continue
            if bs.suite_process is None:
                if self.dry_run:
                    # Dry-run: mark completed immediately
                    bs.status = COMPLETED
                    bs.ended_at = now_iso()
                    bs.exit_code = 0
                    log(f"Block {bid}: [dry-run] COMPLETED")
                    if self.handoff_runtime(bs):
                        continue
                    self._end_runtime(bs)
                continue

            rc = bs.suite_process.poll()
            if rc is None:
                continue  # still running

            # Close log file
            log_file = bs.config.get("_log_file")
            if log_file:
                try:
                    log_file.close()
                except Exception:
                    pass

            bs.ended_at = now_iso()
            bs.exit_code = rc

            # The runtime must still be the verified one when the suite ends; if it was
            # replaced mid-suite, the recorded rows are not this model's.
            changed = None
            if bs.identity and bs.runtime:
                still = served_model_id(bs.runtime.port)
                if still != bs.identity["requested"]:
                    changed = f"port {bs.runtime.port} now serves {still}"

            if changed:
                bs.status = FAILED
                bs.error = f"runtime identity changed during suite ({changed}); results invalid"
                log(f"Block {bid}: FAILED, {bs.error}")
            elif rc == 0:
                bs.status = COMPLETED
                log(f"Block {bid}: COMPLETED (exit 0)")
            else:
                bs.status = FAILED
                bs.error = f"suite exited with code {rc}"
                log(f"Block {bid}: FAILED (exit {rc})")

            if (rc == 0 or self.on_failure == "continue") and self.handoff_runtime(bs):
                continue

            # Unload after the final suite in a group.
            self._end_runtime(bs)

    def shutdown(self) -> None:
        """Stop suites, unload runtimes, cancel what the run still owns in the manager, write final state."""
        self._shutdown = True
        log("Shutting down campaign...")
        for bid, bs in self.state.blocks.items():
            if bs.suite_process and bs.suite_process.poll() is None:
                log(f"  Stopping suite process for {bid}")
                # Stop the suite container
                suite_container = f"bench-{bs.suite.replace('bench-', '')}-campaign-{bid}"
                docker_stop(suite_container)
                try:
                    bs.suite_process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    bs.suite_process.kill()

            self._end_runtime(bs)

            # Close log files
            log_file = bs.config.get("_log_file")
            if log_file:
                try:
                    log_file.close()
                except Exception:
                    pass

            if bs.status in (LOADING, RUNNING_SUITE):
                bs.status = FAILED
                bs.error = "campaign shutdown"
                bs.ended_at = now_iso()

        if self.load_proc is not None and self.load_proc.poll() is None:
            self.load_proc.terminate()  # the request only; the manager's own wait is cancelled below
        self.load_proc = self.loading_block = None
        if not self.dry_run:
            cancel_owner(self.owner, "campaign shutdown")
        self.state.status = "interrupted"
        self.state.ended_at = now_iso()
        write_status(self.state)
        write_checkpoint(self.state)

    def run(self) -> int:
        """Main event loop. Returns 0 if all blocks completed, 1 otherwise."""
        log(f"Campaign: {self.state.name}")
        log(f"Run ID: {self.state.run_id}")
        log(f"Blocks: {len(self.state.blocks)}")
        log(f"Manager owner: {self.owner}")
        if self.dry_run:
            log("MODE: dry-run (no models loaded, no containers started)")
        log("")

        # Print schedule summary
        for bid, bs in self.state.blocks.items():
            if bs.status == COMPLETED:
                log(f"  [{bid}] {bs.model} / {bs.suite} -> ALREADY COMPLETED (checkpoint)")
            else:
                log(f"  [{bid}] {bs.model} / {bs.suite} (cards: {bs.placement})")
        log("")

        while not self._shutdown:
            self.reap_finished()

            if self._all_done():
                break

            if self.on_failure == "stop" and self._any_failed():
                log("Stopping campaign due to block failure (--on-failure stop)")
                self.shutdown()
                break

            self.advance_deps()
            self.check_load_complete()
            self.try_start_load()

            if not self.dry_run:
                write_status(self.state)
                write_checkpoint(self.state)

            if self._all_done():
                break

            if not self.dry_run:
                time.sleep(TICK_INTERVAL)

        if not self.dry_run and not self._shutdown:
            cancel_owner(self.owner, "campaign finished")  # nothing should remain; this makes sure

        # Final status
        completed = sum(1 for bs in self.state.blocks.values() if bs.status == COMPLETED)
        failed = sum(1 for bs in self.state.blocks.values() if bs.status == FAILED)
        total = len(self.state.blocks)

        if failed == 0:
            self.state.status = "completed"
        elif completed > 0:
            self.state.status = "completed_with_failures"
        else:
            self.state.status = "failed"

        self.state.ended_at = now_iso()
        if not self.dry_run:
            write_status(self.state)
            write_checkpoint(self.state)

        log("")
        log("=" * 50)
        log(f"CAMPAIGN {'COMPLETE' if failed == 0 else 'FINISHED WITH FAILURES'}")
        log(f"  Completed: {completed}/{total}")
        if failed:
            log(f"  Failed: {failed}/{total}")
            for bid, bs in self.state.blocks.items():
                if bs.status == FAILED:
                    log(f"    {bid}: {bs.error}")
        log(f"  Status: {HISTORY_ROOT / self.state.name / self.state.run_id / 'status.json'}")
        log("=" * 50)

        return 0 if failed == 0 else 1


# ---------------------------------------------------------------------------
# Manifest Parsing
# ---------------------------------------------------------------------------

def parse_manifest(path: Path) -> tuple[str, Dict, List[Dict]]:
    """Parse a campaign manifest file. Returns (name, defaults, blocks)."""
    data = json.loads(path.read_text(encoding="utf-8"))

    name = data.get("name")
    if not name:
        raise SystemExit(f"ERROR: manifest missing 'name' field: {path}")

    defaults = data.get("defaults", {})
    blocks = data.get("blocks", [])

    if not blocks:
        raise SystemExit(f"ERROR: manifest has no blocks: {path}")

    # Validate blocks
    seen_ids = set()
    for i, block in enumerate(blocks):
        block_id = block.get("id")
        if not block_id:
            raise SystemExit(f"ERROR: block {i} missing 'id' field")
        if block_id in seen_ids:
            raise SystemExit(f"ERROR: duplicate block id '{block_id}'")
        seen_ids.add(block_id)

        if "model" not in block:
            raise SystemExit(f"ERROR: block '{block_id}' missing 'model' field")
        if "suite" not in block:
            raise SystemExit(f"ERROR: block '{block_id}' missing 'suite' field")
        if block["suite"] not in SUPPORTED_SUITES:
            raise SystemExit(
                f"ERROR: block '{block_id}' has unsupported suite '{block['suite']}'. "
                f"Supported: {sorted(SUPPORTED_SUITES)}"
            )

        # Every suite runs against a model the manager loads; the registry names its GGUF.
        if block["suite"] in STANDALONE_SUITES:
            raise SystemExit(
                f"ERROR: block '{block_id}': standalone suite {block['suite']} would need cards without a "
                f"manager load, which the runner no longer takes")
        try:
            block_gpus(block)
        except ValueError as exc:
            raise SystemExit(f"ERROR: block '{block_id}': {exc}")

        # Validate depends_on references
        for dep in block.get("depends_on", []):
            if dep not in seen_ids and dep not in {b.get("id") for b in blocks}:
                raise SystemExit(
                    f"ERROR: block '{block_id}' depends_on '{dep}' which doesn't exist"
                )

    return name, defaults, blocks


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

_VERBOSE = False


def main() -> None:
    global _VERBOSE

    parser = argparse.ArgumentParser(
        description="Unified benchmark campaign runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "manifest",
        type=Path,
        help="Path to campaign manifest JSON",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Run ID for checkpoint/resume (default: timestamp)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print schedule without starting any containers",
    )
    parser.add_argument(
        "--on-failure",
        choices=["continue", "stop"],
        default="continue",
        help="What to do when a block fails (default: continue)",
    )
    parser.add_argument(
        "--max-active-models",
        type=int,
        default=None,
        metavar="N",
        help="Limit concurrently active model/runtime groups (use 1 for sequential models)",
    )
    parser.add_argument(
        "--limit-override",
        type=int,
        default=None,
        metavar="N",
        help="Override the limit for ALL blocks (e.g. --limit-override 10 for a smoke run)",
    )
    parser.add_argument(
        "--limit",
        action="append",
        default=[],
        metavar="BLOCK_ID=N",
        help="Override limit for a specific block (repeatable, e.g. --limit e4b_reasoning=50 --limit e2b_reasoning=50)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print detailed commands and debug info",
    )
    args = parser.parse_args()
    if args.max_active_models is not None and args.max_active_models < 1:
        parser.error("--max-active-models must be at least 1")
    _VERBOSE = args.verbose

    manifest_path = args.manifest.resolve()
    if not manifest_path.exists():
        raise SystemExit(f"ERROR: manifest not found: {manifest_path}")

    load_suite_contracts()

    # Parse manifest
    name, defaults, blocks_raw = parse_manifest(manifest_path)

    # Generate run_id
    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")

    # Parse per-block limit overrides (--limit BLOCK_ID=N)
    per_block_limits: Dict[str, int] = {}
    for spec in args.limit:
        if "=" not in spec:
            raise SystemExit(f"ERROR: --limit must be BLOCK_ID=N, got: {spec}")
        bid, val = spec.split("=", 1)
        try:
            per_block_limits[bid] = int(val)
        except ValueError:
            raise SystemExit(f"ERROR: --limit value must be integer, got: {spec}")

    # Resolve block configs
    if args.limit_override is not None:
        log(f"Limit override: all blocks will use --limit {args.limit_override}")
    if per_block_limits:
        for bid, val in per_block_limits.items():
            log(f"Limit override: {bid} will use --limit {val}")

    block_states: Dict[str, BlockState] = {}
    for block_cfg in blocks_raw:
        block_cfg = resolve_block_config(dict(block_cfg), defaults)
        bid = block_cfg["id"]
        # Per-block --limit wins over --limit-override wins over manifest
        effective_limit = per_block_limits.get(bid)
        if effective_limit is None and args.limit_override is not None:
            effective_limit = args.limit_override
        if effective_limit is not None:
            block_cfg["limit"] = effective_limit
            # Strip any explicit --limit from suite_args so resolve_limit uses our override
            sa = block_cfg.get("suite_args", [])
            cleaned = []
            skip_next = False
            for arg in sa:
                if skip_next:
                    skip_next = False
                    continue
                if arg == "--limit":
                    skip_next = True
                    continue
                cleaned.append(arg)
            block_cfg["suite_args"] = cleaned
        block_cfg["_campaign_name"] = name
        block_cfg["_run_id"] = run_id
        bid = block_cfg["id"]
        block_states[bid] = BlockState(
            block_id=bid,
            config=block_cfg,
        )

    # Build campaign state
    state = CampaignState(
        name=name,
        run_id=run_id,
        manifest_path=str(manifest_path),
        blocks=block_states,
        started_at=now_iso(),
    )

    # Validate deployment compatibility before any model load. Dry runs perform
    # the preflight but never consume or create resumable checkpoints.
    validate_resolved_blocks(block_states)
    preflight_suite_images({bs.suite for bs in block_states.values()})
    preflight_output_roots({bs.suite for bs in block_states.values()})
    if not args.dry_run:
        load_checkpoint(state)

    # Set up signal handlers
    scheduler = Scheduler(
        state=state,
        on_failure=args.on_failure,
        dry_run=args.dry_run,
        max_active_models=args.max_active_models,
    )

    def signal_handler(signum: int, frame: Any) -> None:
        sig_name = signal.Signals(signum).name
        log(f"Received {sig_name}, shutting down...")
        scheduler.shutdown()
        sys.exit(1)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Run
    exit_code = scheduler.run()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
