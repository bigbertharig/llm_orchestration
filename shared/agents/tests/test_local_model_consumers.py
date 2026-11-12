"""Model tools use orchestration's imported files without gateway offer files."""

import sys
import unittest
from pathlib import Path


SHARED = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SHARED / "scripts"))
sys.path.insert(0, str(SHARED / "scripts" / "llama_runtime"))

import manage_model_hotset  # noqa: E402
import smoke_runtime  # noqa: E402


class LocalModelConsumerTests(unittest.TestCase):
    def test_hotset_uses_local_active_list_and_explicit_candidate(self):
        active = manage_model_hotset.selected_gguf_paths()
        self.assertIn("qwen3.6:27b", active)
        self.assertNotIn("qwen3.8-flash-next:q2_0", active)
        selected = manage_model_hotset.selected_gguf_paths(["qwen3.8-flash-next:q2_0"])
        self.assertEqual(list(selected), ["qwen3.8-flash-next:q2_0"])

    def test_smoke_uses_imported_roles_and_active_list(self):
        models = smoke_runtime.models_in_use()
        self.assertIn("qwen3.6:27b", models)
        self.assertIn("qwen3.6:35b-a3b-xl", models)
        self.assertEqual(len(models), len(set(models)))


if __name__ == "__main__":
    unittest.main()
