#!/usr/bin/env python3
"""Tests for controller-neutral GPU leases."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from gpu_leases import (
    GPULeaseConflict,
    GPULeaseNotFound,
    GPULeaseOwnershipError,
    GPULeaseStateError,
    GPULeaseStore,
)


class GPULeaseStoreTests(unittest.TestCase):
    def test_atomic_group_acquire_and_lookup(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = GPULeaseStore(tmp)
            lease = store.acquire(
                controller="plans",
                owner="gpu-1",
                gpu_ids=[3, 1],
                run_id="plan-pool",
                ports=[11437, 11435],
            )

            self.assertEqual(lease["gpu_ids"], [1, 3])
            self.assertEqual(lease["controller"], "plans")
            self.assertEqual(store.for_gpu(1)["lease_id"], lease["lease_id"])
            self.assertEqual(store.for_gpu(3)["lease_id"], lease["lease_id"])
            self.assertIsNone(store.for_gpu(2))

    def test_overlap_rejects_entire_group(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = GPULeaseStore(tmp)
            store.acquire(
                controller="supervision",
                owner="session-a",
                gpu_ids=[0],
                run_id="county-a",
            )

            with self.assertRaises(GPULeaseConflict):
                store.acquire(
                    controller="plans",
                    owner="brain",
                    gpu_ids=[0, 1],
                    run_id="plan-pool",
                )
            self.assertIsNone(store.for_gpu(1))

    def test_expired_lease_still_blocks_until_explicit_release(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = GPULeaseStore(tmp)
            lease = store.acquire(
                controller="benchmarks",
                owner="campaign",
                gpu_ids=[2],
                run_id="run-1",
            )
            state = store.read_state()
            state["leases"][0]["expires_at"] = (
                datetime.now(timezone.utc) - timedelta(seconds=1)
            ).isoformat()
            store.path.write_text(json.dumps(state), encoding="utf-8")

            with self.assertRaisesRegex(GPULeaseConflict, "expired_requires_reconciliation"):
                store.acquire(
                    controller="plans",
                    owner="gpu-2",
                    gpu_ids=[2],
                    run_id="plan-pool",
                )

            store.release(lease["lease_id"], owner="operator", force=True)
            replacement = store.acquire(
                controller="plans",
                owner="gpu-2",
                gpu_ids=[2],
                run_id="plan-pool",
            )
            self.assertEqual(replacement["controller"], "plans")

    def test_renew_and_release_require_owner(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = GPULeaseStore(tmp)
            lease = store.acquire(
                controller="interactive",
                owner="operator-a",
                gpu_ids=[4],
                run_id="chat-1",
            )

            with self.assertRaises(GPULeaseOwnershipError):
                store.renew(lease["lease_id"], owner="operator-b")
            renewed = store.renew(lease["lease_id"], owner="operator-a")
            self.assertEqual(renewed["generation"], lease["generation"])
            with self.assertRaises(GPULeaseOwnershipError):
                store.release(lease["lease_id"], owner="operator-b")
            store.release(lease["lease_id"], owner="operator-a")
            self.assertIsNone(store.get(lease["lease_id"]))

    def test_missing_release_fails_loud(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = GPULeaseStore(tmp)
            with self.assertRaises(GPULeaseNotFound):
                store.release("missing", owner="operator")

    def test_invalid_or_duplicate_persisted_state_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = GPULeaseStore(tmp)
            store.path.parent.mkdir(parents=True)
            store.path.write_text("not json", encoding="utf-8")
            with self.assertRaises(GPULeaseStateError):
                store.for_gpu(1)

        with tempfile.TemporaryDirectory() as tmp:
            store = GPULeaseStore(tmp)
            store.path.parent.mkdir(parents=True)
            store.path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "revision": 2,
                        "updated_at": "",
                        "leases": [
                            {"lease_id": "one", "gpu_ids": [1]},
                            {"lease_id": "two", "gpu_ids": [1]},
                        ],
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaises(GPULeaseStateError):
                store.for_gpu(1)


if __name__ == "__main__":
    unittest.main()
