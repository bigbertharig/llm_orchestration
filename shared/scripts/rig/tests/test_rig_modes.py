import sys
import unittest
import json
import tempfile
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

AGENTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AGENTS))
import rig  # noqa: E402


class ModeTests(unittest.TestCase):
    def test_boot_stops_managed_runtimes_before_recording_idle(self):
        scan = {"ok": True, "visible": [], "ids": [], "errors": [], "changes": []}
        with patch.object(rig, "Lock", return_value=nullcontext()), \
             patch.object(rig, "stop_everything") as stop, \
             patch.object(rig, "gpu_check", return_value=scan), \
             patch.object(rig, "write_state"), \
             patch.object(rig, "hold_plan_for_boot"):
            self.assertEqual(rig.cmd_boot(None), 0)
        stop.assert_called_once_with()

    def test_lease_preview_uses_live_admission_without_writing_a_lease(self):
        cards = [{"index": 1, "total_mb": 6144, "used_mb": 0},
                 {"index": 2, "total_mb": 8192, "used_mb": 0}]
        state = Mock()
        state.leases.return_value = []
        state.claims.return_value = []
        args = SimpleNamespace(action="preview", spec=json.dumps({
            "gpus_needed": [{"min_vram_mb": 5000},
                            {"min_vram_mb": 5000, "max_vram_mb": 7000}]}))
        with patch.object(rig, "open_hw_state", return_value=state), \
             patch.object(rig, "open_plan") as plan, \
             patch.object(rig.hw_state, "FileLock", return_value=nullcontext()), \
             patch.object(rig, "gpu_check", return_value={"ok": True, "visible": cards}), \
             patch.object(rig, "llama_containers", return_value=[]), \
             patch.object(rig, "live_ports", return_value={}), \
             patch.object(rig, "emit", side_effect=lambda payload, code=0: payload):
            preview = rig.cmd_lease(args)
            refused = rig.cmd_lease(SimpleNamespace(action="preview", spec=json.dumps({
                "gpus_needed": [{"min_vram_mb": 30000}]})))
            malformed = rig.cmd_lease(SimpleNamespace(action="preview", spec="null"))
        self.assertEqual(preview, {"admitted": True, "gpu_ids": [2, 1],
                                   "ports": [11435, 11436], "port": 11435})
        self.assertEqual(refused["kind"], "capacity")
        self.assertIn("no capacity", refused["error"])
        self.assertEqual(malformed["kind"], "invalid")
        state.acquire.assert_not_called()
        plan.assert_not_called()

    def test_lease_admission_finds_complete_fit_for_bounded_cards(self):
        cards = [{"index": 1, "total_mb": 6144, "used_mb": 0},
                 {"index": 2, "total_mb": 8192, "used_mb": 0}]
        state = Mock()
        state.leases.return_value = []
        state.claims.return_value = []
        with patch.object(rig, "gpu_check", return_value={"ok": True, "visible": cards}), \
             patch.object(rig, "llama_containers", return_value=[]), \
             patch.object(rig, "live_ports", return_value={}):
            chosen = rig.admit_lease_spec({"gpus_needed": [{"min_vram_mb": 5000},
                                                         {"min_vram_mb": 5000, "max_vram_mb": 7000}]}, state)
        self.assertEqual(chosen["gpu_ids"], [2, 1])

    def test_lease_admission_uses_live_smallest_fit_and_returns_specific_capacity(self):
        cards = [{"index": 0, "total_mb": 24576, "used_mb": 0},
                 {"index": 2, "total_mb": 8192, "used_mb": 0},
                 {"index": 5, "total_mb": 6144, "used_mb": 0}]
        state = Mock()
        state.leases.return_value = []
        state.claims.return_value = []
        with patch.object(rig, "gpu_check", return_value={"ok": True, "visible": cards}), \
             patch.object(rig, "llama_containers", return_value=[]), \
             patch.object(rig, "live_ports", return_value={}):
            chosen = rig.admit_lease_spec({"gpus_needed": [{"min_vram_mb": 20000},
                                                         {"min_vram_mb": 5000}]}, state)
            self.assertEqual(chosen["gpu_ids"], [0, 5])
            self.assertEqual(chosen["ports"], [11434, 11439])
            self.assertEqual(chosen["metadata"]["port"], 11434)
            with self.assertRaisesRegex(rig.hw_state.StateError, "no capacity") as refused:
                rig.admit_lease_spec({"gpus_needed": [{"min_vram_mb": 30000}]}, state)
            self.assertEqual(refused.exception.kind, "capacity")

    def test_lease_admission_rejects_busy_pinned_card_and_preserves_claim(self):
        cards = [{"index": 2, "total_mb": 8192, "used_mb": 0}]
        state = Mock()
        state.leases.return_value = []
        state.claims.return_value = [{"name": "bench", "gpu_ids": [2]}]
        with patch.object(rig, "gpu_check", return_value={"ok": True, "visible": cards}), \
             patch.object(rig, "llama_containers", return_value=[]), \
             patch.object(rig, "live_ports", return_value={}):
            with self.assertRaisesRegex(rig.hw_state.StateError, "claimed by bench"):
                rig.admit_lease_spec({"gpu_ids": [2]}, state)
            chosen = rig.admit_lease_spec({"gpu_ids": [2], "metadata": {"claim": "bench"}}, state)
            self.assertEqual(chosen["gpu_ids"], [2])

    def test_scan_saves_live_card_identity_without_a_fixed_card_count(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "hardware_inventory.json"
            cards = [{"index": 3, "uuid": "GPU-example", "name": "Example GPU",
                      "total_mb": 6144, "bus": "01:00.0", "link_gen": 3,
                      "link_width": 8, "root_port": None, "via": []}]
            scan = {"ok": True, "visible": cards, "errors": [], "changes": [],
                    "since_boot": [], "config_not_present": [], "ids": [3]}
            with patch.object(rig, "INVENTORY_PATH", path), patch.object(rig, "gpu_check", return_value=scan):
                self.assertEqual(rig.cmd_scan(None), 0)
            saved = json.loads(path.read_text())
            self.assertEqual(saved["schema"], "rig-hardware-inventory/1")
            self.assertEqual(saved["cards"][0]["uuid"], "GPU-example")
            self.assertEqual(saved["cards"][0]["total_mb"], 6144)

    def test_use_picks_the_load_mode(self):
        self.assertEqual(rig.load_mode({"use": "benchmark"}), "bench")
        self.assertEqual(rig.load_mode({"use": "chat"}), "remote")
        self.assertEqual(rig.load_mode({"use": "worker"}), "remote")
        self.assertEqual(rig.load_mode({"id": "published"}), "remote")

    def test_one_mode_at_a_time(self):
        self.assertFalse(rig.mode_blocks("idle", "bench", force=False))
        self.assertFalse(rig.mode_blocks("bench", "bench", force=False))
        self.assertTrue(rig.mode_blocks("remote", "bench", force=False))
        self.assertTrue(rig.mode_blocks("bench", "remote", force=False))
        self.assertTrue(rig.mode_blocks("orchestration", "bench", force=False))
        # force always stops whatever is active first
        self.assertTrue(rig.mode_blocks("bench", "bench", force=True))
        self.assertFalse(rig.mode_blocks("idle", "remote", force=True))

    def test_partial_ready_requires_exact_layers_context_and_ram_reserve(self):
        entry = {"n_gpu_layers": 48, "ctx_size": 32768}
        info = {"offloaded": 48, "layers": 89, "n_ctx": 32768}
        self.assertIsNone(rig.partial_ready_error("x", entry, info, 5000))
        self.assertIn("offload mismatch", rig.partial_ready_error("x", entry, {**info, "offloaded": 47}, 5000))
        self.assertIn("context mismatch", rig.partial_ready_error("x", entry, {**info, "n_ctx": 16384}, 5000))
        self.assertIn("4096 MB", rig.partial_ready_error("x", entry, info, 4000))


    def test_only_manager_load_modes_remain(self):
        self.assertEqual(set(rig.MODES), {"bench", "remote"})
        self.assertEqual(set(rig.LOAD_MODES), {"bench", "remote"})

    def test_hotset_prefers_present_local_copy(self):
        import tempfile
        from model_paths import resolve_hotset_gguf
        with tempfile.TemporaryDirectory() as tmp:
            archive = Path("/mnt/shared/models/example/model.gguf")
            hotset = Path(tmp) / "example" / "model.gguf"
            hotset.parent.mkdir()
            self.assertEqual(resolve_hotset_gguf(archive, {"model_search_roots": [tmp]}), archive)
            hotset.write_bytes(b"model")
            self.assertEqual(resolve_hotset_gguf(archive, {"model_search_roots": [tmp]}), hotset)

    def test_remote_load_requires_manager_spec(self):
        with self.assertRaisesRegex(rig.RigError, "manager --spec"):
            rig.cmd_start_remote(SimpleNamespace(spec=None, profiles=["chat"]))


    def test_guard_summary_marks_stale_owner_state_unknown(self):
        with patch.object(rig.hw_guard, "admission", return_value=(False, "unknown", "stale")), \
             patch.object(rig.hw_guard, "read_state", return_value={
                 "updated_at": "2026-11-25T00:00:00Z",
                 "sample": {"gpu_temp_max_c": 90}}):
            summary = rig.guard_summary()
        self.assertEqual(summary["schema"], "rig-hardware-telemetry/1")
        self.assertFalse(summary["fresh"])
        self.assertIsNone(summary["sampled_at"])
        self.assertEqual(summary["level"], "unknown")

    def test_guard_sample_time_is_utc_from_epoch(self):
        with patch.object(rig.hw_guard, "admission", return_value=(True, "ok", "ok")), \
             patch.object(rig.hw_guard, "read_state", return_value={
                 "updated_at": "2026-11-25T17:30:00",
                 "updated_ts": 1795653000,
                 "sample": {"gpu_temp_max_c": 53}}):
            summary = rig.guard_summary()
        self.assertTrue(summary["fresh"])
        self.assertEqual(summary["sampled_at"], "2026-11-26T00:30:00+00:00")

    def test_standalone_status_exposes_owner_guard(self):
        state = {"mode": "idle", "status": "idle", "profiles": {}}
        guard = {"schema": "rig-hardware-telemetry/1", "level": "ok", "fresh": True}
        scan = {"visible": [], "ids": [], "ok": True, "errors": []}
        with patch.object(rig, "read_state", return_value=state), \
             patch.object(rig, "gpu_check", return_value=scan), \
             patch.object(rig, "gpu_notes", return_value=[]), \
             patch.object(rig, "guard_summary", return_value=guard), \
             patch.object(Path, "exists", return_value=False):
            status = rig.status_payload()
        self.assertEqual(status["guard"], guard)


if __name__ == "__main__":
    unittest.main()
