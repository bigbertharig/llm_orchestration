import sys
import json
import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

RIG = Path(__file__).resolve().parents[2] / "scripts" / "rig"
sys.path.insert(0, str(RIG))
import kv_chat_control  # noqa: E402
import model_modes  # noqa: E402


class WorkerInventoryTests(unittest.TestCase):
    def test_scans_capability_and_prefers_smaller_cards(self):
        output = ("0, NVIDIA GeForce RTX 3090 Ti, 24564, 19\n"
                  "2, NVIDIA GeForce GTX 1070, 8192, 6\n"
                  "3, NVIDIA GeForce GTX 1060 6GB, 6144, 4\n"
                  "6, NVIDIA GeForce GTX 1060 6GB, 6144, 4\n"
                  "7, NVIDIA GeForce GTX 1050, 4096, 4\n")
        with patch.object(kv_chat_control, "run", return_value=SimpleNamespace(stdout=output)):
            cards = kv_chat_control.worker_cards()
        self.assertEqual([card["gpu"] for card in cards], [3, 6, 2])

    def test_catalog_uses_scanned_candidates(self):
        cards = [{"gpu": 3}, {"gpu": 6}, {"gpu": 2}]
        with patch.object(kv_chat_control, "worker_cards", return_value=cards):
            row = next(item for item in model_modes.catalog()
                       if item["kind"] == "attached_chat")
        self.assertEqual(row["candidate_gpus"], [3, 6, 2])
        self.assertTrue(row["executable"])


class ClaimLifecycleTests(unittest.TestCase):
    def test_chat_claim_is_bounded(self):
        with patch.object(kv_chat_control, "controller", return_value="pi"), \
             patch.object(kv_chat_control, "run") as run:
            kv_chat_control.claim("chat", 2, "test")
        argv = run.call_args.args[0]
        self.assertEqual(argv[argv.index("--ttl") + 1], str(kv_chat_control.CLAIM_TTL_S))

    def test_running_service_renews_each_claim(self):
        st = {"instance_id": "current", "claims": {"main": "main", "chat": "worker"}}
        with patch.object(kv_chat_control, "locked", return_value=nullcontext()), \
             patch.object(kv_chat_control, "state", return_value=st), \
             patch.object(kv_chat_control, "container_running", return_value=True), \
             patch.object(kv_chat_control, "controller", return_value="pi"), \
             patch.object(kv_chat_control, "run") as run:
            self.assertTrue(kv_chat_control.renew_claims_once("current"))
        self.assertEqual([c.args[0][3] for c in run.call_args_list], ["main", "worker"])

    def test_dead_service_drops_claims_and_state(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.json"
            path.write_text(json.dumps({"instance_id": "current", "claims": {"main": "main"}}))
            with patch.object(kv_chat_control, "STATE", path), \
                 patch.object(kv_chat_control, "locked", return_value=nullcontext()), \
                 patch.object(kv_chat_control, "container_running", return_value=False), \
                 patch.object(kv_chat_control, "release") as release:
                self.assertFalse(kv_chat_control.renew_claims_once("current"))
            release.assert_called_once_with("main")
            self.assertFalse(path.exists())


if __name__ == "__main__":
    unittest.main()
