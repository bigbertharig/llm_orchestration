#!/usr/bin/env python3
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))

if "filelock" not in sys.modules or not hasattr(sys.modules["filelock"], "Timeout"):
    sys.modules["filelock"] = types.SimpleNamespace(FileLock=object, Timeout=Exception)
if "requests" not in sys.modules:
    sys.modules["requests"] = types.SimpleNamespace()

import gpu_tasks
import gpu_split
from gpu import GPUAgent
from gpu_split import GPUSplitMixin
from gpu_tasks import GPUTaskMixin
from gpu_workers import GPUWorkerMixin


class DummyLock:
    def __init__(self, *_args, **_kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class MockSplitGpu(GPUSplitMixin, GPUTaskMixin, GPUWorkerMixin):
    def __init__(self, tmpdir: Path):
        self.name = "gpu-3"
        self.logger = MagicMock()
        self.config = {}
        self.split_state_dir = tmpdir / "split_llm"
        self.split_state_dir.mkdir(parents=True, exist_ok=True)
        self.shared_path = tmpdir
        self.port = 11433
        self.runtime_port = 11441
        self.runtime_group_id = "pair_3_4"
        self.runtime_placement = "split_gpu"
        self.loaded_model = "model-a"
        self.model_loaded = True
        self.loaded_tier = 2
        self.runtime_state = "ready_split"
        self.runtime_error_code = None
        self.runtime_error_detail = None
        self.model_tier_by_id = {"model-a": 2}
        self.split_runtime_owner = False
        self.split_runtime_generation = "gen-a"
        self.split_runtime_owner_meta_path = None
        self.split_runtime_invariant_failures = 0
        self.split_runtime_port_model_miss_timestamps = []
        self.split_runtime_ready_at = None
        self.active_meta_task = None
        self.active_workers = {}
        self.outbox = []
        self.stats = {"tasks_completed": 0, "tasks_failed": 0}
        self.cleanup_calls = []
        self.reported_issues = []
        self.reset_calls = []
        self.kill_calls = []
        self.load_model_calls = []
        self.start_runtime_calls = 0
        self.reuse_existing = False
        self.listener_active = True
        self.runtime_attestation = {
            "ok": True,
            "mismatch_reason": "",
            "details": {"actual_port_models": []},
        }
        self.reservation_path = self._split_reservation_path("pair_3_4")
        self._service_calls = 0

    def _split_reservation_path(self, group_id: str) -> Path:
        return self.split_state_dir / f"{group_id}.json"

    def _touch_meta_task(self, phase="", force=False):
        return None

    def _set_runtime_state(self, new_state, task_id=None, phase=None):
        self.runtime_state = new_state
        self.runtime_transition_task_id = task_id
        self.runtime_transition_phase = phase

    def _get_gpu_config(self, member):
        suffix = int(str(member).split("-")[-1])
        return {"id": suffix}

    def _full_local_reset(
        self,
        reason: str,
        *,
        count_toward_circuit_breaker: bool = True,
    ) -> None:
        self.reset_calls.append(
            {
                "reason": reason,
                "count_toward_circuit_breaker": count_toward_circuit_breaker,
            }
        )
        self.runtime_state = gpu_split.RUNTIME_STATE_COLD
        self.model_loaded = False
        self.loaded_model = None
        self.runtime_placement = "single_gpu"
        self.runtime_group_id = None

    def load_model(self, model_id=None, task_id=None):
        self.load_model_calls.append({"model_id": model_id, "task_id": task_id})
        self.model_loaded = True
        self.loaded_model = model_id
        self.runtime_state = "ready_single"

    def start_runtime_legacy(self):
        self.start_runtime_calls += 1

    def _atomic_join_split_reservation(self, reservation, _group_id):
        return reservation, True, ""

    def _is_group_member(self, group):
        members = [str(m).strip() for m in group.get("members", []) if str(m).strip()]
        return self.name in members

    def _service_split_reservations(self):
        self._service_calls += 1
        if self._service_calls == 1 and self.reservation_path.exists():
            self.reservation_path.unlink()

    def _coordinated_split_failure_cleanup(self, group_id, split_port, reason, task_id=None):
        payload = {
            "group_id": group_id,
            "split_port": split_port,
            "reason": reason,
            "task_id": task_id,
        }
        self.cleanup_calls.append(payload)
        return payload

    def _runtime_reset_port_and_state(
        self,
        ports_to_clean,
        reason,
        task_id=None,
        *,
        stop_split_runtime=False,
        stop_local_runtime=False,
    ):
        payload = {
            "ports_to_clean": ports_to_clean,
            "reason": reason,
            "task_id": task_id,
            "stop_split_runtime": stop_split_runtime,
            "stop_local_runtime": stop_local_runtime,
        }
        self.cleanup_calls.append(payload)
        return payload

    def _split_group_has_any_active_work(self, _group_id: str) -> bool:
        return False

    def _read_json_file(self, path: Path):
        if not path.exists():
            return None
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _split_runtime_has_expected_owner(self, port, model_id, group_id=None):
        return self.reuse_existing

    def _split_runtime_has_any_listener(self, port):
        return True

    def _split_runtime_has_model_loaded(self, port, model_id):
        return self.reuse_existing

    def _split_runtime_has_any_listener(self, port):
        return self.listener_active

    def _kill_local_listener_on_port(self, port):
        self.listener_active = False
        self.cleanup_calls.append({"port": port, "reason": "listener_reclaimed"})
        return 1

    def _report_split_health_issue(self, **kwargs):
        self.reported_issues.append(kwargs)

    def _clear_split_health_issue(self):
        return None

    def _attest_runtime_reality(self):
        return self.runtime_attestation

    def _kill_worker(self, worker_id: str, reason: str = "", force: bool = False):
        self.kill_calls.append(
            {"worker_id": worker_id, "reason": reason, "force": force}
        )


class MockLoadClaimGpu(GPUTaskMixin):
    def __init__(self):
        self.name = "gpu-2"
        self.logger = MagicMock()
        self.model_loaded = False
        self.loaded_tier = 0
        self.runtime_state = "wedged"
        self.runtime_placement = "single_gpu"
        self.model_tier_by_id = {"model-a": 2}
        self.reset_calls = []

    def _can_accept_load_task(self):
        if self.runtime_state == "wedged":
            return False, "wedged_requires_reclaim"
        return True, ""

    def _full_local_reset(
        self,
        reason: str,
        *,
        count_toward_circuit_breaker: bool = True,
    ) -> None:
        self.reset_calls.append(
            {
                "reason": reason,
                "count_toward_circuit_breaker": count_toward_circuit_breaker,
            }
        )
        self.runtime_state = "cold"
        self.runtime_placement = "single_gpu"
        self.model_loaded = False

    def _is_in_split_pair_loading_lock(self):
        return False, None

    def _format_split_pair_lock_reason(self, _lock_info):
        return "split_pair_loading"

    def _is_group_member(self, group):
        members = [str(m).strip() for m in group.get("members", []) if str(m).strip()]
        return self.name in members


class SplitRuntimeHardeningTests(unittest.TestCase):
    def setUp(self):
        self._orig_lock = gpu_tasks.FileLock
        self._orig_split_lock = gpu_split.FileLock
        gpu_tasks.FileLock = DummyLock
        gpu_split.FileLock = DummyLock

    def tearDown(self):
        gpu_tasks.FileLock = self._orig_lock
        gpu_split.FileLock = self._orig_split_lock

    def test_load_split_reservation_disappearance_forces_cleanup(self):
        with tempfile.TemporaryDirectory() as tmp:
            gpu = MockSplitGpu(Path(tmp))
            task = {
                "task_id": "task-1",
                "target_model": "model-a",
                "candidate_groups": [
                    {"id": "pair_3_4", "members": ["gpu-3", "gpu-4"], "port": 11441}
                ],
            }

            result = gpu._execute_meta_command(task, "load_split_llm")

            self.assertFalse(result["success"])
            self.assertEqual(result["error"], "Split load failed: reservation disappeared")
            self.assertIn("failure_reason=reservation disappeared", result["diagnostic"])
            self.assertIn("reservation disappeared", result["output"])
            self.assertEqual(len(gpu.cleanup_calls), 1)
            self.assertEqual(
                gpu.cleanup_calls[0]["reason"],
                "dead_split_state:reservation_disappeared",
            )

    def test_owner_loss_invariant_forces_immediate_reset(self):
        with tempfile.TemporaryDirectory() as tmp:
            gpu = MockSplitGpu(Path(tmp))
            gpu.reservation_path.write_text(
                json.dumps(
                    {
                        "group_id": "pair_3_4",
                        "status": "ready",
                        "target_model": "model-a",
                        "members": ["gpu-3", "gpu-4"],
                        "port": 11441,
                    }
                ),
                encoding="utf-8",
            )

            gpu._check_split_runtime_invariants()

            self.assertEqual(len(gpu.cleanup_calls), 1)
            self.assertEqual(
                gpu.cleanup_calls[0]["reason"],
                "dead_split_state:owner_meta_mismatch",
            )
            self.assertEqual(gpu.reported_issues, [])

    def test_pending_split_coordination_detects_partner_nudge(self):
        with tempfile.TemporaryDirectory() as tmp:
            gpu = MockSplitGpu(Path(tmp))
            gpu._write_split_partner_nudge("pair_3_4", ["gpu-3", "gpu-4"], "gpu-3")

            self.assertTrue(gpu._has_pending_split_coordination())

    def test_gpu_loop_sleep_shortens_for_split_coordination(self):
        agent = GPUAgent.__new__(GPUAgent)
        agent._has_pending_split_coordination = lambda: True

        self.assertEqual(agent._loop_sleep_seconds(), 0.5)

        agent._has_pending_split_coordination = lambda: False
        self.assertEqual(agent._loop_sleep_seconds(), 5)

    def test_split_warmup_request_budget_is_bounded(self):
        gpu = MockSplitGpu(Path(tempfile.mkdtemp()))
        original_time = gpu_split.time.time
        try:
            gpu_split.time.time = lambda: 100.0
            self.assertEqual(gpu._split_warmup_request_budget_seconds(500.0), 25)
            self.assertEqual(gpu._split_warmup_request_budget_seconds(110.0), 10)
        finally:
            gpu_split.time.time = original_time

    def test_split_runtime_start_budget_reserves_warmup_tail(self):
        gpu = MockSplitGpu(Path(tempfile.mkdtemp()))

        self.assertEqual(gpu._split_runtime_start_budget_seconds(), 270)

    def test_split_runtime_start_budget_honors_profile_timeout(self):
        gpu = MockSplitGpu(Path(tempfile.mkdtemp()))
        gpu.config = {
            "llama_split_profiles": {
                "qwen2.5-coder:14b": {
                    "meta_timeout_seconds": 600,
                    "extra_args": ["--no-warmup"],
                }
            }
        }

        timeout_seconds = gpu._split_meta_timeout_seconds(model_id="qwen2.5-coder:14b")

        self.assertEqual(timeout_seconds, 600)
        self.assertEqual(
            gpu._split_runtime_start_budget_seconds(meta_timeout_seconds=timeout_seconds),
            570,
        )

    def test_split_warmup_failure_classifier_marks_terminal_failures(self):
        gpu = MockSplitGpu(Path(tempfile.mkdtemp()))

        self.assertTrue(gpu._split_warmup_failure_is_terminal(status_code=499))
        self.assertTrue(
            gpu._split_warmup_failure_is_terminal(
                error_text="timed out waiting for llama runner to start: context canceled"
            )
        )
        self.assertTrue(
            gpu._split_warmup_failure_is_terminal(
                error_text="HTTP 500 read timed out while warming model"
            )
        )
        self.assertFalse(
            gpu._split_warmup_failure_is_terminal(
                status_code=500,
                error_text="temporary warmup http 500",
            )
        )

    def test_load_llm_claim_auto_resets_wedged_runtime(self):
        gpu = MockLoadClaimGpu()

        can_claim, reason = gpu._can_claim_meta_task(
            {
                "command": "load_llm",
                "target_model": "qwen2.5:7b",
                "load_mode": "single",
                "candidate_workers": ["gpu-2"],
            }
        )

        self.assertTrue(can_claim)
        self.assertEqual(reason, "")
        self.assertEqual(len(gpu.reset_calls), 1)
        self.assertEqual(gpu.reset_calls[0]["reason"], "auto_reclaim_before_load_llm")
        self.assertFalse(gpu.reset_calls[0]["count_toward_circuit_breaker"])

    def test_load_split_claim_auto_resets_wedged_runtime(self):
        gpu = MockLoadClaimGpu()

        can_claim, reason = gpu._can_claim_meta_task(
            {
                "command": "load_split_llm",
                "target_model": "model-a",
                "candidate_groups": [
                    {"id": "pair_2_3", "members": ["gpu-2", "gpu-3"], "port": 11440}
                ],
            }
        )

        self.assertTrue(can_claim)
        self.assertEqual(reason, "")
        self.assertEqual(len(gpu.reset_calls), 1)
        self.assertEqual(gpu.reset_calls[0]["reason"], "auto_reclaim_before_load_split_llm")
        self.assertFalse(gpu.reset_calls[0]["count_toward_circuit_breaker"])

    def test_split_runtime_stable_model_presence_accepts_loaded_model(self):
        gpu = MockSplitGpu(Path(tempfile.mkdtemp()))
        original_time = gpu_split.time.time
        original_sleep = gpu_split.time.sleep
        original_requests = gpu_split.requests

        class Response:
            def __init__(self, models):
                self.status_code = 200
                self._models = models

            def json(self):
                return {"models": [{"name": name} for name in self._models]}

        gpu.loaded_model = "qwen2.5-coder:14b"
        ticks = iter([0, 1, 2, 3, 4, 5])
        gpu_split.time.time = lambda: next(ticks, 999)
        gpu_split.time.sleep = lambda _s: None
        gpu_split.requests = types.SimpleNamespace(
            get=lambda _url, timeout=5: Response(["qwen2.5-coder:14b"])
        )
        try:
            ok = gpu._split_runtime_has_stable_model_presence(11440, "qwen2.5-coder:14b")
            self.assertTrue(ok)
        finally:
            gpu_split.time.time = original_time
            gpu_split.time.sleep = original_sleep
            gpu_split.requests = original_requests

    def test_split_join_resets_dirty_member_before_join(self):
        with tempfile.TemporaryDirectory() as tmp:
            gpu = MockSplitGpu(Path(tmp))
            gpu.runtime_state = "ready_single"
            gpu.model_loaded = True
            gpu.loaded_model = "qwen2.5:7b"
            original_requests = gpu_split.requests
            gpu_split.requests = types.SimpleNamespace(
                get=lambda _url, timeout=2: types.SimpleNamespace(status_code=503),
                exceptions=types.SimpleNamespace(ConnectionError=Exception),
            )

            try:
                precond = gpu._ensure_local_split_member_clean_precondition(
                    "pair_3_4",
                    allow_rejoin_group=False,
                )
            finally:
                gpu_split.requests = original_requests

            self.assertTrue(precond["ok"])
            self.assertEqual(len(gpu.reset_calls), 1)
            self.assertEqual(gpu.reset_calls[0]["reason"], "pre_split_join:runtime_not_cold:ready_single")
            self.assertFalse(gpu.reset_calls[0]["count_toward_circuit_breaker"])
            self.assertTrue(precond["details"]["reset_performed"])
            self.assertEqual(precond["details"]["initial_reason_code"], "runtime_not_cold:ready_single")

    def test_split_join_resets_even_clean_member_before_join(self):
        with tempfile.TemporaryDirectory() as tmp:
            gpu = MockSplitGpu(Path(tmp))
            gpu.runtime_state = gpu_split.RUNTIME_STATE_COLD
            gpu.model_loaded = False
            gpu.loaded_model = None
            gpu.runtime_placement = "single_gpu"
            gpu.runtime_group_id = None
            original_requests = gpu_split.requests
            gpu_split.requests = types.SimpleNamespace(
                get=lambda _url, timeout=2: types.SimpleNamespace(status_code=503),
                exceptions=types.SimpleNamespace(ConnectionError=Exception),
            )

            try:
                precond = gpu._ensure_local_split_member_clean_precondition(
                    "pair_3_4",
                    allow_rejoin_group=False,
                )
            finally:
                gpu_split.requests = original_requests

            self.assertTrue(precond["ok"])
            self.assertEqual(len(gpu.reset_calls), 1)
            self.assertEqual(gpu.reset_calls[0]["reason"], "pre_split_join:always_reset")
            self.assertFalse(gpu.reset_calls[0]["count_toward_circuit_breaker"])
            self.assertEqual(precond["details"]["initial_reason_code"], "")
            self.assertTrue(precond["details"]["reset_performed"])

    def test_split_join_backfill_uses_active_runtime_without_reset(self):
        with tempfile.TemporaryDirectory() as tmp:
            gpu = MockSplitGpu(Path(tmp))
            gpu.runtime_state = gpu_split.RUNTIME_STATE_LOADING_SPLIT
            gpu.runtime_placement = "split_gpu"
            gpu.runtime_group_id = "pair_3_4"
            gpu.model_loaded = True
            gpu.loaded_model = "model-a"

            reservation = {
                "group_id": "pair_3_4",
                "status": "ready_stabilizing",
                "target_model": "model-a",
                "members": ["gpu-3", "gpu-4"],
                "joined": {"gpu-3": {"joined_at": "now"}},
                "member_clean": {},
            }

            updated, success, reason = GPUSplitMixin._atomic_join_split_reservation(
                gpu,
                reservation,
                "pair_3_4",
            )

            self.assertTrue(success)
            self.assertEqual(reason, "")
            self.assertEqual(gpu.reset_calls, [])
            self.assertIn("gpu-3", updated["member_clean"])
            self.assertTrue(
                updated["member_clean"]["gpu-3"]["details"]["backfilled_from_active_runtime"]
            )

    def test_service_split_reservations_does_not_rejoin_during_loading(self):
        class LoadingJoinGuardGpu(MockSplitGpu):
            def __init__(self, tmpdir: Path):
                super().__init__(tmpdir)
                self.join_calls = 0

            def _join_split_reservation(self, reservation, reservation_path):
                self.join_calls += 1
                return reservation

        with tempfile.TemporaryDirectory() as tmp:
            gpu = LoadingJoinGuardGpu(Path(tmp))
            gpu.runtime_state = gpu_split.RUNTIME_STATE_LOADING_SPLIT
            reservation_path = gpu._split_reservation_path("pair_3_4")
            reservation_path.write_text(
                json.dumps(
                    {
                        "group_id": "pair_3_4",
                        "status": "loading",
                        "target_model": "model-a",
                        "members": ["gpu-3", "gpu-4"],
                        "launcher": "gpu-3",
                        "port": 11441,
                        "joined": {
                            "gpu-3": {"joined_at": "now"},
                            "gpu-4": {"joined_at": "now"},
                        },
                        "member_clean": {
                            "gpu-3": {"verified_at": "now", "details": {}},
                            "gpu-4": {"verified_at": "now", "details": {}},
                        },
                        "prepared": {
                            "gpu-3": {"prepared_at": "now"},
                            "gpu-4": {"prepared_at": "now"},
                        },
                    }
                ),
                encoding="utf-8",
            )

            gpu._service_split_reservations()

            self.assertEqual(gpu.join_calls, 0)

    def test_service_split_reservations_does_not_restart_runtime_while_loading(self):
        class LoadingRestartGuardGpu(MockSplitGpu):
            def __init__(self, tmpdir: Path):
                super().__init__(tmpdir)
                self.start_calls = 0

            def _start_split_runtime(self, group, model_id, task_id=None):
                self.start_calls += 1
                return True, False

            def _is_gpu_heartbeat_fresh(self, _gpu_name, max_age_seconds=0):
                return True

            def _split_runtime_has_model_loaded(self, port, model_id):
                return False

        with tempfile.TemporaryDirectory() as tmp:
            gpu = LoadingRestartGuardGpu(Path(tmp))
            gpu.runtime_state = gpu_split.RUNTIME_STATE_LOADING_SPLIT
            gpu.active_meta_task = {"task_id": "task-1"}
            reservation_path = gpu._split_reservation_path("pair_3_4")
            reservation_path.write_text(
                json.dumps(
                    {
                        "group_id": "pair_3_4",
                        "status": "loading",
                        "target_model": "model-a",
                        "members": ["gpu-3", "gpu-4"],
                        "launcher": "gpu-3",
                        "port": 11441,
                        "joined": {
                            "gpu-3": {"joined_at": "now"},
                            "gpu-4": {"joined_at": "now"},
                        },
                        "member_clean": {
                            "gpu-3": {"verified_at": "now", "details": {}},
                            "gpu-4": {"verified_at": "now", "details": {}},
                        },
                        "prepared": {
                            "gpu-3": {"prepared_at": "now"},
                            "gpu-4": {"prepared_at": "now"},
                        },
                    }
                ),
                encoding="utf-8",
            )

            gpu._service_split_reservations()

            self.assertEqual(gpu.start_calls, 0)

    def test_active_split_partner_loss_fails_local_llm_workers_and_resets(self):
        with tempfile.TemporaryDirectory() as tmp:
            gpu = MockSplitGpu(Path(tmp))
            reservation_path = gpu._split_reservation_path("pair_3_4")
            reservation_path.write_text(
                json.dumps(
                    {
                        "group_id": "pair_3_4",
                        "status": "ready",
                        "target_model": "model-a",
                        "members": ["gpu-3", "gpu-4"],
                        "launcher": "gpu-3",
                        "port": 11441,
                    }
                ),
                encoding="utf-8",
            )
            gpu.active_workers = {
                "gpu-3-w0-task": {
                    "process": MagicMock(),
                    "task": {
                        "task_id": "task-1",
                        "task_class": "llm",
                    },
                    "vram_estimate": 256,
                    "peak_vram_mb": 128,
                }
            }
            gpu.claimed_vram = 256
            gpu._is_gpu_heartbeat_fresh = lambda _gpu_name, max_age_seconds=0: False

            gpu._check_active_split_partner_health()

            self.assertEqual(len(gpu.kill_calls), 1)
            self.assertEqual(len(gpu.outbox), 1)
            self.assertEqual(gpu.outbox[0].result["error_code"], "split_partner_lost")
            self.assertEqual(gpu.cleanup_calls[-1]["reason"], "split_partner_lost:heartbeat_stale:gpu-4")
            self.assertEqual(gpu.claimed_vram, 0)
            self.assertEqual(gpu.stats["tasks_failed"], 1)

    def test_service_split_reservations_ignores_runtime_owner_metadata_files(self):
        class RuntimeOwnerScanGuardGpu(MockSplitGpu):
            def __init__(self, tmpdir: Path):
                super().__init__(tmpdir)
                self.join_calls = 0

            def _join_split_reservation(self, reservation, reservation_path):
                self.join_calls += 1
                return reservation

        with tempfile.TemporaryDirectory() as tmp:
            gpu = RuntimeOwnerScanGuardGpu(Path(tmp))
            (gpu.split_state_dir / "pair_3_4.runtime_owner.json").write_text(
                json.dumps(
                    {
                        "group_id": "pair_3_4",
                        "launcher": "gpu-3",
                        "members": ["gpu-3", "gpu-4"],
                        "model_id": "model-a",
                        "port": 11441,
                    }
                ),
                encoding="utf-8",
            )

            gpu._service_split_reservations()

            self.assertEqual(gpu.join_calls, 0)
            self.assertFalse(gpu._has_pending_split_coordination())

    def test_split_runtime_reuses_existing_healthy_listener(self):
        with tempfile.TemporaryDirectory() as tmp:
            gpu = MockSplitGpu(Path(tmp))
            gpu.reuse_existing = True
            gpu.runtime_backend = "llama"
            gpu.config = {}
            gpu.worker_num_ctx = 4096

            ok, cleanup_done = gpu._start_split_runtime(
                {
                    "id": "pair_3_4",
                    "members": ["gpu-3", "gpu-4"],
                    "port": 11441,
                },
                "qwen2.5-coder:14b",
                task_id="task-1",
            )

            self.assertTrue(ok)
            self.assertFalse(cleanup_done)
            self.assertEqual(gpu.cleanup_calls, [])
            self.assertIn("SPLIT_RUNTIME_REUSE_EXISTING", str(gpu.logger.info.call_args_list))

    def test_reclaim_stale_split_listener_before_launch_removes_named_container(self):
        with tempfile.TemporaryDirectory() as tmp:
            gpu = MockSplitGpu(Path(tmp))
            original_run = gpu_split.subprocess.run

            def fake_run(cmd, capture_output=True, text=True, timeout=15):
                self.assertEqual(cmd, ["docker", "rm", "-f", "llama-split-pair_3_4"])
                gpu.listener_active = False
                return types.SimpleNamespace(returncode=0, stdout="", stderr="")

            gpu_split.subprocess.run = fake_run
            try:
                reclaimed = gpu._reclaim_stale_split_listener_before_launch(
                    port=11441,
                    container_name="llama-split-pair_3_4",
                    group_id="pair_3_4",
                )
            finally:
                gpu_split.subprocess.run = original_run

            self.assertTrue(reclaimed)
            self.assertFalse(gpu.listener_active)
            self.assertEqual(gpu.cleanup_calls, [])
            self.assertIn(
                "SPLIT_RUNTIME_RECLAIM_CONTAINER",
                str(gpu.logger.warning.call_args_list),
            )

    def test_split_join_fails_when_reset_does_not_restore_clean_state(self):
        class StubbornSplitGpu(MockSplitGpu):
            def _full_local_reset(
                self,
                reason: str,
                *,
                count_toward_circuit_breaker: bool = True,
            ) -> None:
                self.reset_calls.append(
                    {
                        "reason": reason,
                        "count_toward_circuit_breaker": count_toward_circuit_breaker,
                    }
                )
                self.runtime_state = "ready_single"
                self.model_loaded = True
                self.loaded_model = "still-dirty"

        with tempfile.TemporaryDirectory() as tmp:
            gpu = StubbornSplitGpu(Path(tmp))
            gpu.runtime_state = "ready_single"
            gpu.model_loaded = True
            gpu.loaded_model = "qwen2.5:7b"

            precond = gpu._ensure_local_split_member_clean_precondition(
                "pair_3_4",
                allow_rejoin_group=False,
            )

            self.assertFalse(precond["ok"])
            self.assertTrue(precond["reason_code"].startswith("reset_failed:"))
            self.assertEqual(len(gpu.reset_calls), 1)
            self.assertFalse(gpu.reset_calls[0]["count_toward_circuit_breaker"])

    def test_load_llm_always_resets_before_loading(self):
        with tempfile.TemporaryDirectory() as tmp:
            gpu = MockSplitGpu(Path(tmp))
            gpu.runtime_state = "ready_single"
            gpu.model_loaded = True
            gpu.loaded_model = "old-model"

            result = gpu._execute_meta_command(
                {"task_id": "task-2", "target_model": "new-model"},
                "load_llm",
            )

            self.assertTrue(result["success"])
            self.assertEqual(len(gpu.reset_calls), 1)
            self.assertEqual(gpu.reset_calls[0]["reason"], "pre_load_llm:new-model")
            self.assertFalse(gpu.reset_calls[0]["count_toward_circuit_breaker"])
            self.assertEqual(gpu.start_runtime_calls, 0)
            self.assertEqual(len(gpu.load_model_calls), 1)
            self.assertEqual(gpu.load_model_calls[0]["model_id"], "new-model")
            self.assertEqual(gpu.loaded_model, "new-model")
            self.assertEqual(result.get("error_code", ""), "")

    def test_load_llm_missing_target_model_fails_loud(self):
        with tempfile.TemporaryDirectory() as tmp:
            gpu = MockSplitGpu(Path(tmp))

            result = gpu._execute_meta_command(
                {"task_id": "task-3"},
                "load_llm",
            )

            self.assertFalse(result["success"])
            self.assertEqual(result["error_code"], "missing_target_model")
            self.assertEqual(gpu.load_model_calls, [])

    def test_load_meta_attestation_mismatch_allows_execution_cleanup(self):
        with tempfile.TemporaryDirectory() as tmp:
            gpu = MockSplitGpu(Path(tmp))
            gpu.runtime_attestation = {
                "ok": False,
                "mismatch_reason": "port_probe:model_present",
                "details": {"actual_port_models": ["qwen2.5:7b"]},
            }

            load_llm = gpu._attest_meta_task_precondition({"command": "load_llm"})
            load_split = gpu._attest_meta_task_precondition({"command": "load_split_llm"})

            self.assertFalse(load_llm["ok"])
            self.assertTrue(load_llm["allow_continue"])
            self.assertEqual(load_llm["mismatch_reason"], "port_probe:model_present")
            self.assertFalse(load_split["ok"])
            self.assertTrue(load_split["allow_continue"])
            self.assertEqual(load_split["mismatch_reason"], "port_probe:model_present")


if __name__ == "__main__":
    unittest.main()
