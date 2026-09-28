import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "model_access_api.py"
SPEC = importlib.util.spec_from_file_location("model_access_api", SCRIPT)
model_access_api = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = model_access_api
SPEC.loader.exec_module(model_access_api)


class FakeLeaseStore:
    def __init__(self):
        self.leases = {}
        self.renewed = []
        self.released = []

    def get(self, lease_id):
        return self.leases.get(lease_id)

    def renew(self, lease_id, *, owner, ttl_seconds):
        self.renewed.append((lease_id, owner, ttl_seconds))
        return self.leases[lease_id]

    def release(self, lease_id, *, owner):
        self.released.append((lease_id, owner))
        self.leases.pop(lease_id)


class ModelAccessServiceTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tempdir.cleanup)
        self.catalog = Path(self.tempdir.name) / "models.catalog.json"
        self.access_config = Path(self.tempdir.name) / "model_access.json"
        self.catalog.write_text(
            json.dumps(
                {
                    "models": [
                        {
                            "id": "qwen-coder",
                            "placement": "single_gpu",
                            "tier": 1,
                            "tags": ["code"],
                        },
                        {
                            "id": "brain-only",
                            "placement": "brain_gpu",
                            "tier": 3,
                        },
                    ]
                }
            ),
            encoding="utf-8",
        )
        self.access_config.write_text(
            json.dumps({"schema_version": 1, "models": ["qwen-coder"]}),
            encoding="utf-8",
        )
        self.store = FakeLeaseStore()
        self.loads = []

        def loader(model_id, owner, ttl):
            self.loads.append((model_id, owner, ttl))
            lease = {
                "lease_id": "interactive-lease-1",
                "owner": owner,
                "gpu_ids": [2],
            }
            self.store.leases[lease["lease_id"]] = lease
            return {
                "ok": True,
                "model": model_id,
                "runtime_model": "Qwen-Coder.gguf",
                "port": 11436,
                "lease": lease,
                "split_group": None,
            }

        self.service = model_access_api.ModelAccessService(
            catalog_path=self.catalog,
            access_config_path=self.access_config,
            lease_store=self.store,
            runtime_loader=loader,
            rig_host="10.0.0.3",
        )

    def test_lists_only_worker_models(self):
        self.assertEqual(
            self.service.list_models(),
            [
                {
                    "id": "qwen-coder",
                    "placement": "single_gpu",
                    "tier": 1,
                    "tags": ["code"],
                }
            ],
        )

    def test_acquire_returns_runtime_endpoint_and_session(self):
        result = self.service.acquire("qwen-coder", 900)

        self.assertEqual(result["endpoint"], "http://10.0.0.3:11436/v1")
        self.assertEqual(result["runtime_model"], "Qwen-Coder.gguf")
        self.assertEqual(result["placement"]["gpus"], [2])
        self.assertTrue(result["session_id"].startswith("interactive-"))
        self.assertFalse(result["reused"])
        self.assertEqual(len(self.loads), 1)

    def test_same_model_sessions_share_one_allocation(self):
        first = self.service.acquire("qwen-coder", 900)
        second = self.service.acquire("qwen-coder", 900)

        self.assertEqual(len(self.loads), 1)
        self.assertTrue(second["reused"])
        self.service.release(first["session_id"])
        self.assertEqual(self.store.released, [])
        self.service.release(second["session_id"])
        self.assertEqual(len(self.store.released), 1)

    def test_renew_updates_canonical_lease(self):
        acquired = self.service.acquire("qwen-coder", 900)
        renewed = self.service.renew(acquired["session_id"], 1200)

        self.assertEqual(renewed["session_id"], acquired["session_id"])
        self.assertEqual(self.store.renewed[-1][2], 1200)

    def test_unknown_model_fails_without_loading(self):
        with self.assertRaises(model_access_api.ModelNotFound):
            self.service.acquire("missing", 900)
        self.assertEqual(self.loads, [])


if __name__ == "__main__":
    unittest.main()
