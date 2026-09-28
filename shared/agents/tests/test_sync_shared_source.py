import importlib.util
import os
import stat
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "sync_shared_source.py"
SPEC = importlib.util.spec_from_file_location("sync_shared_source", SCRIPT)
sync_shared_source = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = sync_shared_source
SPEC.loader.exec_module(sync_shared_source)


class SyncSharedSourceTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tempdir.cleanup)
        self.root = Path(self.tempdir.name)

    def test_protected_paths_are_excluded(self):
        self.assertTrue(sync_shared_source.is_protected("shared/core/RULES.md"))
        self.assertTrue(
            sync_shared_source.is_protected(
                "shared/agents/permissions/worker.json"
            )
        )
        self.assertFalse(
            sync_shared_source.is_protected("shared/agents/gpu_leases.py")
        )

    def test_deploy_file_preserves_content_and_mode(self):
        source = self.root / "source.py"
        destination = self.root / "live" / "source.py"
        source.write_text("print('current')\n", encoding="utf-8")
        source.chmod(0o750)

        sync_shared_source.deploy_one(source, destination)

        self.assertEqual(destination.read_text(encoding="utf-8"), "print('current')\n")
        self.assertEqual(stat.S_IMODE(destination.stat().st_mode), 0o750)
        self.assertEqual(
            sync_shared_source.compare_path(source, destination),
            "same",
        )

    def test_deploy_symlink_replaces_existing_file(self):
        source = self.root / "source-link"
        destination = self.root / "live" / "source-link"
        source.symlink_to("../catalog/models.json")
        destination.parent.mkdir(parents=True)
        destination.write_text("stale\n", encoding="utf-8")

        sync_shared_source.deploy_one(source, destination)

        self.assertTrue(destination.is_symlink())
        self.assertEqual(os.readlink(destination), "../catalog/models.json")
        self.assertEqual(
            sync_shared_source.compare_path(source, destination),
            "same",
        )


if __name__ == "__main__":
    unittest.main()
