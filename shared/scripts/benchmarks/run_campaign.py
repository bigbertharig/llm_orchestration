#!/usr/bin/env python3
"""Unified benchmark campaign runner.

Reads a declarative manifest (model + suite + settings per block),
auto-schedules across all available GPUs in parallel, and manages the
full lifecycle: load model -> run suite -> reap -> advance.

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
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TICK_INTERVAL = 3  # seconds between scheduler ticks
SHARED_ROOT = Path(os.environ.get("CAMPAIGN_SHARED_ROOT", "/mnt/shared"))
RUNTIME_SCRIPT = SHARED_ROOT / "scripts" / "llama_runtime" / "run_runtime.sh"
TUNING_PROFILES_PATH = SHARED_ROOT / "plans" / "shoulders" / "benchmarking" / "model_tuning_profiles.json"
SUITE_CONTRACTS_PATH = SHARED_ROOT / "plans" / "shoulders" / "benchmarking" / "docker" / "suite_contracts.json"
MODELS_CATALOG_PATH = SHARED_ROOT / "agents" / "models.catalog.json"
HISTORY_ROOT = SHARED_ROOT / "logs" / "benchmarks" / "campaigns" / "history"
BENCHMARKING_SCRIPTS = SHARED_ROOT / "plans" / "shoulders" / "benchmarking"
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
# GPU Slot Model
# ---------------------------------------------------------------------------

@dataclass
class GpuSlot:
    name: str
    gpu_ids: List[int]
    port: int
    tier: str  # brain, single, split

    @property
    def gpu_set(self) -> Set[int]:
        return set(self.gpu_ids)


# Static slot definitions matching the rig layout
SLOTS: Dict[str, GpuSlot] = {
    "brain": GpuSlot(name="brain", gpu_ids=[0], port=11434, tier="brain"),
    "gpu_1": GpuSlot(name="gpu_1", gpu_ids=[1], port=11435, tier="single"),
    "gpu_2": GpuSlot(name="gpu_2", gpu_ids=[2], port=11436, tier="single"),
    "gpu_3": GpuSlot(name="gpu_3", gpu_ids=[3], port=11437, tier="single"),
    "gpu_4": GpuSlot(name="gpu_4", gpu_ids=[4], port=11438, tier="single"),
    "gpu_5": GpuSlot(name="gpu_5", gpu_ids=[5], port=11439, tier="single"),
    "split_1_3": GpuSlot(name="split_1_3", gpu_ids=[1, 3], port=11435, tier="split"),
    "split_4_5": GpuSlot(name="split_4_5", gpu_ids=[4, 5], port=11438, tier="split"),
}

# Placement -> candidate slot names (preference order)
PLACEMENT_SLOTS: Dict[str, List[str]] = {
    "brain": ["brain"],
    "single": ["gpu_1", "gpu_2", "gpu_3", "gpu_4", "gpu_5"],
    "split_1_3": ["split_1_3"],
    "split_4_5": ["split_4_5"],
}


def slots_conflict(a: GpuSlot, b: GpuSlot) -> bool:
    """Two slots conflict if their GPU sets overlap."""
    return bool(a.gpu_set & b.gpu_set)


def find_free_slot(
    placement: str, occupied: Dict[str, str]
) -> Optional[GpuSlot]:
    """Find a free slot for the given placement.

    occupied: slot_name -> block_id currently using it.
    Returns the first non-conflicting slot, or None.
    """
    candidates = PLACEMENT_SLOTS.get(placement)
    if not candidates:
        return None

    occupied_slots = [SLOTS[name] for name in occupied if name in SLOTS]

    for slot_name in candidates:
        slot = SLOTS[slot_name]
        conflict = any(slots_conflict(slot, occ) for occ in occupied_slots)
        if not conflict:
            return slot
    return None


# ---------------------------------------------------------------------------
# Block State
# ---------------------------------------------------------------------------

@dataclass
class BlockState:
    block_id: str
    config: Dict[str, Any]
    status: str = PENDING
    assigned_slot: Optional[GpuSlot] = None
    runtime_container: Optional[str] = None
    suite_process: Optional[subprocess.Popen] = None
    suite_log_path: Optional[Path] = None
    started_at: Optional[str] = None
    load_started_at: Optional[str] = None
    suite_started_at: Optional[str] = None
    ended_at: Optional[str] = None
    exit_code: Optional[int] = None
    error: Optional[str] = None

    @property
    def placement(self) -> str:
        return self.config.get("placement", "single")

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
    """Stop and remove a Docker container by name."""
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


def probe_health(port: int, timeout: int = 5) -> bool:
    """Check if a llama-server is responding on the given port."""
    try:
        import urllib.request
        url = f"http://127.0.0.1:{port}/v1/models"
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8", errors="replace"))
            models = data.get("data", [])
            return len(models) > 0
    except Exception:
        return False


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


def resolve_block_config(block: Dict, defaults: Dict) -> Dict:
    """Merge block config with campaign defaults and tuning profile lookups."""
    model_id = block["model"]
    tuning = _lookup_tuning(model_id) or {}
    tuning_runtime = tuning.get("runtime", {})

    # ctx_size: block > tuning profile > 2048
    if "ctx_size" not in block:
        block["ctx_size"] = tuning_runtime.get("ctx_size", 2048)

    # batch_size: block > tuning profile > 128
    if "batch_size" not in block:
        block["batch_size"] = tuning_runtime.get("batch_size", 128)

    # runtime_image: block > defaults > fallback
    if "runtime_image" not in block:
        block["runtime_image"] = defaults.get("runtime_image", "llama-runtime:sm61-sm86")

    # load_timeout_s: block > defaults > 300
    if "load_timeout_s" not in block:
        block["load_timeout_s"] = defaults.get("load_timeout_s", 300)

    # runtime_args: block list + tuning extra_args merged
    block_rt_args = list(block.get("runtime_args", []))
    tuning_extra = tuning.get("extra_args", [])
    if isinstance(tuning_extra, list):
        # Only add tuning extras not already present in block args
        existing = set(block_rt_args)
        for arg in tuning_extra:
            if arg not in existing:
                block_rt_args.append(arg)
    block["runtime_args"] = block_rt_args

    # depends_on: default empty
    if "depends_on" not in block:
        block["depends_on"] = []

    # placement: default single
    if "placement" not in block:
        block["placement"] = "single"

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
    if errors:
        raise SystemExit("ERROR: campaign preflight failed:\n  - " + "\n  - ".join(errors))


# ---------------------------------------------------------------------------
# Runtime Layer (wraps run_runtime.sh)
# ---------------------------------------------------------------------------

def build_runtime_cmd(block: BlockState, slot: GpuSlot) -> List[str]:
    """Build the shell command to start a llama-server container."""
    cfg = block.config
    container_name = f"llama-campaign-{block.block_id}"

    gpu_spec = ",".join(str(g) for g in slot.gpu_ids)

    cmd = [
        "bash", str(RUNTIME_SCRIPT),
        "--name", container_name,
        "--model", cfg["gguf"],
        "--port", str(slot.port),
        "--gpus", f"device={gpu_spec}",
        "--image", cfg["runtime_image"],
        "--ctx-size", str(cfg["ctx_size"]),
        "--batch-size", str(cfg.get("batch_size", 128)),
    ]

    # Memory limits based on tier
    # With mmap (default in run_runtime.sh), GGUF file pages are reclaimable
    # under cgroup pressure. Anonymous memory scales with ctx_size (KV cache
    # scratch buffers), not model size:
    #   single (ctx 4096):  peaks ~0.8-1.6 GB anon
    #   brain  (ctx 16384): peaks ~8-9 GB anon
    #   split:              similar to brain (large ctx for 14B models)
    if slot.tier == "brain":
        cmd.extend(["--memory-limit", "10g", "--memory-swap", "12g"])
    elif slot.tier == "split":
        cmd.extend(["--memory-limit", "10g", "--memory-swap", "12g"])
    else:
        cmd.extend(["--memory-limit", "2g", "--memory-swap", "3g"])

    # Tensor split for multi-GPU
    if len(slot.gpu_ids) > 1:
        # Even split across GPUs
        ts = ",".join(["1"] * len(slot.gpu_ids))
        cmd.extend(["--tensor-split", ts])

    # Extra runtime args (e.g., --reasoning-budget 0)
    for arg in cfg.get("runtime_args", []):
        cmd.extend(["--extra-arg", str(arg)])

    return cmd


def start_runtime(block: BlockState, slot: GpuSlot) -> str:
    """Start the runtime container. Returns container name."""
    container_name = f"llama-campaign-{block.block_id}"

    # Stop any leftover container with the same name
    docker_stop(container_name)
    time.sleep(1)

    cmd = build_runtime_cmd(block, slot)
    log(f"  Loading runtime: {container_name}")
    log(f"  Command: {' '.join(cmd)}", verbose_only=True)

    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        stderr = result.stderr.strip()
        raise RuntimeError(
            f"Failed to start runtime for {block.block_id}: "
            f"exit {result.returncode}, stderr: {stderr}"
        )

    return container_name


def wait_for_runtime(port: int, timeout_s: int = 300) -> bool:
    """Poll until the runtime responds on the given port, or timeout."""
    start = time.monotonic()
    while time.monotonic() - start < timeout_s:
        if probe_health(port):
            return True
        time.sleep(5)
    return False


def stop_runtime(container_name: Optional[str]) -> None:
    """Stop a runtime container."""
    if container_name:
        log(f"  Stopping runtime: {container_name}")
        docker_stop(container_name)


# ---------------------------------------------------------------------------
# Suite Layer (wraps docker run bench-*)
# ---------------------------------------------------------------------------

def build_suite_cmd(block: BlockState, slot: GpuSlot) -> List[str]:
    """Build the docker run command for a benchmark suite."""
    cfg = block.config
    suite = cfg["suite"]
    spec = SUITE_SPECS[suite]
    container_name = f"bench-{suite.replace('bench-', '')}-campaign-{block.block_id}"
    model_id = cfg["model"]
    port = slot.port
    run_id = cfg.get("_run_id", "unknown")
    run_name = f"campaign_{block.block_id}_{run_id}"
    suite_log_root = suite_results_root(suite)

    # Resolve limit
    extra_limit_args = resolve_limit(cfg)
    suite_args = list(cfg.get("suite_args", [])) + extra_limit_args
    run_class = str(cfg["run_class"])
    gpu_spec = ",".join(str(g) for g in slot.gpu_ids)
    hardware_id = str(
        cfg.get("hardware_id")
        or f"rig-{slot.name}-gpu{'-'.join(str(g) for g in slot.gpu_ids)}"
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
            "-v", f"{SHARED_ROOT}:{SHARED_ROOT}",
            "-v", f"{suite_log_root}:/results",
            "-v", f"{BENCHMARKING_SCRIPTS}:/benchmark-scripts:ro",
        ]
        if spec.get("container_gpu_access"):
            cmd.extend(["--gpus", f"device={gpu_spec}"])
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
            cmd.extend(["--gpu-ids", gpu_spec])
        cmd.extend(suite_args)

    else:
        raise ValueError(f"Unsupported suite: {suite}")

    return cmd


def start_suite(block: BlockState, slot: GpuSlot) -> subprocess.Popen:
    """Launch the suite container as a subprocess."""
    cmd = build_suite_cmd(block, slot)
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
        if bs.assigned_slot:
            entry["gpu_slot"] = bs.assigned_slot.name
            entry["port"] = bs.assigned_slot.port
        entry["suite"] = bs.suite
        if bs.started_at:
            entry["started_at"] = bs.started_at
        if bs.ended_at:
            entry["ended_at"] = bs.ended_at
        if bs.exit_code is not None:
            entry["exit_code"] = bs.exit_code
        if bs.error:
            entry["error"] = bs.error
        if bs.suite_log_path:
            entry["log_path"] = str(bs.suite_log_path)

        # Provide waiting_for info
        if bs.status == WAITING_GPU:
            # Find who's using the needed slot
            for obid, obs in state.blocks.items():
                if obid == bid:
                    continue
                if obs.assigned_slot and obs.status in (LOADING, RUNNING_SUITE):
                    slot_needed = PLACEMENT_SLOTS.get(bs.placement, [])
                    if obs.assigned_slot.name in slot_needed or any(
                        slots_conflict(SLOTS[sn], obs.assigned_slot)
                        for sn in slot_needed
                        if sn in SLOTS
                    ):
                        entry["waiting_for"] = f"{obs.assigned_slot.name} ({obid} {obs.status})"
                        break
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
            "slot": bs.assigned_slot.name if bs.assigned_slot else None,
            "runtime_container": bs.runtime_container,
            "started_at": bs.started_at,
            "ended_at": bs.ended_at,
            "exit_code": bs.exit_code,
            "error": bs.error,
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
            # Clean up orphan container
            orphan = saved.get("runtime_container")
            if orphan:
                docker_stop(orphan)

    return True


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------

class Scheduler:
    def __init__(
        self,
        state: CampaignState,
        on_failure: str = "continue",
        dry_run: bool = False,
    ):
        self.state = state
        self.on_failure = on_failure
        self.dry_run = dry_run
        self.load_lock = False  # only one model loads at a time
        self.loading_block: Optional[str] = None
        self.occupied: Dict[str, str] = {}  # slot_name -> block_id
        self._shutdown = False

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
        """Pick a ready block and start loading its model (if load_lock is free)."""
        if self.load_lock:
            return

        for bid, bs in self.state.blocks.items():
            if bs.status != WAITING_GPU:
                continue

            slot = find_free_slot(bs.placement, self.occupied)
            if slot is None:
                continue

            # Check if this suite is standalone (bench-knowledge) — no runtime needed
            if bs.suite in STANDALONE_SUITES:
                # No load phase — go straight to suite
                bs.assigned_slot = slot
                bs.status = RUNNING_SUITE
                bs.started_at = now_iso()
                bs.suite_started_at = now_iso()
                self.occupied[slot.name] = bid
                log(f"Block {bid} -> {slot.name} (port {slot.port}) [standalone suite, no runtime load]")

                if not self.dry_run:
                    try:
                        proc = start_suite(bs, slot)
                        bs.suite_process = proc
                    except Exception as e:
                        bs.status = FAILED
                        bs.error = str(e)
                        bs.ended_at = now_iso()
                        self.occupied.pop(slot.name, None)
                        log(f"Block {bid}: FAILED to start suite: {e}")
                return

            # Worker suite — need to load runtime first
            bs.assigned_slot = slot
            bs.status = LOADING
            bs.started_at = now_iso()
            bs.load_started_at = now_iso()
            self.occupied[slot.name] = bid
            self.load_lock = True
            self.loading_block = bid
            log(f"Block {bid} -> {slot.name} (port {slot.port}) [loading model]")

            if self.dry_run:
                log(f"  Runtime command: {' '.join(build_runtime_cmd(bs, slot))}", verbose_only=True)
            else:
                try:
                    container_name = start_runtime(bs, slot)
                    bs.runtime_container = container_name
                except Exception as e:
                    bs.status = FAILED
                    bs.error = f"runtime start failed: {e}"
                    bs.ended_at = now_iso()
                    self.load_lock = False
                    self.loading_block = None
                    self.occupied.pop(slot.name, None)
                    log(f"Block {bid}: FAILED to start runtime: {e}")
            return  # only start one load per tick

    def check_load_complete(self) -> None:
        """Check if the currently loading block's runtime is ready."""
        if not self.load_lock or not self.loading_block:
            return

        bs = self.state.blocks[self.loading_block]
        if bs.status != LOADING:
            # Something else changed the status (e.g., shutdown)
            self.load_lock = False
            self.loading_block = None
            return

        if self.dry_run:
            # In dry-run, instantly "complete" the load
            bs.status = RUNNING_SUITE
            bs.suite_started_at = now_iso()
            self.load_lock = False
            self.loading_block = None
            log(f"Block {bs.block_id}: [dry-run] model loaded, starting suite")
            if bs.assigned_slot:
                log(
                    f"  Suite command: {' '.join(build_suite_cmd(bs, bs.assigned_slot))}",
                    verbose_only=True,
                )
            return

        slot = bs.assigned_slot
        if not slot:
            return

        timeout_s = bs.config.get("load_timeout_s", 300)
        load_start = bs.load_started_at or bs.started_at or now_iso()
        # Parse the ISO timestamp
        try:
            start_dt = datetime.fromisoformat(load_start)
            if start_dt.tzinfo is None:
                start_dt = start_dt.replace(tzinfo=timezone.utc)
            elapsed = (datetime.now(timezone.utc) - start_dt).total_seconds()
        except (ValueError, TypeError):
            elapsed = 0

        if probe_health(slot.port):
            log(f"Block {bs.block_id}: model ready on port {slot.port} ({elapsed:.0f}s)")
            bs.status = RUNNING_SUITE
            bs.suite_started_at = now_iso()
            self.load_lock = False
            self.loading_block = None

            try:
                proc = start_suite(bs, slot)
                bs.suite_process = proc
            except Exception as e:
                bs.status = FAILED
                bs.error = f"suite start failed: {e}"
                bs.ended_at = now_iso()
                stop_runtime(bs.runtime_container)
                self.occupied.pop(slot.name, None)
                log(f"Block {bs.block_id}: FAILED to start suite: {e}")

        elif elapsed > timeout_s:
            log(f"Block {bs.block_id}: TIMEOUT loading model after {elapsed:.0f}s")
            bs.status = FAILED
            bs.error = f"load timeout ({timeout_s}s)"
            bs.ended_at = now_iso()
            stop_runtime(bs.runtime_container)
            self.load_lock = False
            self.loading_block = None
            self.occupied.pop(slot.name, None)

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
                    if bs.assigned_slot:
                        self.occupied.pop(bs.assigned_slot.name, None)
                    stop_runtime(bs.runtime_container)
                    log(f"Block {bid}: [dry-run] COMPLETED")
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

            if rc == 0:
                bs.status = COMPLETED
                log(f"Block {bid}: COMPLETED (exit 0)")
            else:
                bs.status = FAILED
                bs.error = f"suite exited with code {rc}"
                log(f"Block {bid}: FAILED (exit {rc})")

            # Release slot and stop runtime
            if bs.assigned_slot:
                self.occupied.pop(bs.assigned_slot.name, None)
            stop_runtime(bs.runtime_container)

    def release_slot(self, block_id: str) -> None:
        """Force-release a slot."""
        bs = self.state.blocks.get(block_id)
        if bs and bs.assigned_slot:
            self.occupied.pop(bs.assigned_slot.name, None)

    def shutdown(self) -> None:
        """Stop all running containers and write final state."""
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

            if bs.runtime_container:
                stop_runtime(bs.runtime_container)

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

        self.state.status = "interrupted"
        self.state.ended_at = now_iso()
        write_status(self.state)
        write_checkpoint(self.state)

    def run(self) -> int:
        """Main event loop. Returns 0 if all blocks completed, 1 otherwise."""
        log(f"Campaign: {self.state.name}")
        log(f"Run ID: {self.state.run_id}")
        log(f"Blocks: {len(self.state.blocks)}")
        if self.dry_run:
            log("MODE: dry-run (no containers will be started)")
        log("")

        # Print schedule summary
        for bid, bs in self.state.blocks.items():
            if bs.status == COMPLETED:
                log(f"  [{bid}] {bs.model} / {bs.suite} -> ALREADY COMPLETED (checkpoint)")
            else:
                log(f"  [{bid}] {bs.model} / {bs.suite} (placement={bs.placement})")
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

        # Worker suites need a gguf path
        if block["suite"] in WORKER_SUITES and "gguf" not in block:
            raise SystemExit(f"ERROR: block '{block_id}' (worker suite) missing 'gguf' field")
        if block["suite"] in STANDALONE_SUITES and "gguf" not in block:
            raise SystemExit(f"ERROR: block '{block_id}' (standalone suite) missing 'gguf' field")

        placement = block.get("placement", "single")
        if placement not in PLACEMENT_SLOTS:
            raise SystemExit(
                f"ERROR: block '{block_id}' has invalid placement '{placement}'. "
                f"Valid: {sorted(PLACEMENT_SLOTS.keys())}"
            )

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
