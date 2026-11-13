import sys
import unittest
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


if __name__ == "__main__":
    unittest.main()
