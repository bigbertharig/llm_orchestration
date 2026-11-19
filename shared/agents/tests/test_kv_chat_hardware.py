import sys
import json
import tempfile
import unittest
from contextlib import nullcontext, redirect_stdout
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

RIG = Path(__file__).resolve().parents[2] / "scripts" / "rig"
sys.path.insert(0, str(RIG))
import kv_chat_control  # noqa: E402
import model_modes  # noqa: E402


class WorkerInventoryTests(unittest.TestCase):
    def test_scans_capability_and_prefers_smaller_cards(self):
        output = ("0, NVIDIA GeForce RTX 3090 Ti, 24564, 19, 24094\n"
                  "2, NVIDIA GeForce GTX 1070, 8192, 6, 8102\n"
                  "3, NVIDIA GeForce GTX 1060 6GB, 6144, 4, 6062\n"
                  "6, NVIDIA GeForce GTX 1060 6GB, 6144, 4, 6062\n"
                  "7, NVIDIA GeForce GTX 1050, 4096, 4, 4028\n")
        pci = ("0, 00000000:01:00.0\n2, 00000000:03:00.0\n"
               "3, 00000000:04:00.0\n6, 00000000:05:00.0\n7, 00000000:06:00.0\n")
        with patch.object(kv_chat_control, "run", side_effect=[
                SimpleNamespace(stdout=output), SimpleNamespace(stdout=pci)]):
            cards = kv_chat_control.worker_cards()
        self.assertEqual([card["gpu"] for card in cards], [3, 6, 2])
        self.assertEqual(cards[0]["free_mib"], 6062)

    def test_catalog_uses_scanned_candidates(self):
        cards = [{"gpu": 3}, {"gpu": 6}, {"gpu": 2}]
        with patch.object(kv_chat_control, "worker_cards", return_value=cards):
            row = next(item for item in model_modes.catalog()
                       if item["kind"] == "attached_chat")
        self.assertEqual(row["candidate_gpus"], [3, 6, 2])
        self.assertTrue(row["executable"])

    def test_catalog_distinguishes_main_and_headless_qwen_workers(self):
        cards = [{"gpu": 1}]
        with patch.object(kv_chat_control, "worker_cards", return_value=cards):
            rows = model_modes.catalog()
        pool = next(row for row in rows if row["mode"] == "gpu1060-pool-chat-q4")
        self.assertEqual(pool["requires"]["mode"], "gpu0-worker-pool-q4")
        parent = next(row for row in rows if row["mode"] == "gpu0-worker-pool-q4")
        self.assertEqual(parent["main_context_cap"], 0)

    def test_nemo_offers_model_cards_and_maps_physical_to_runtime_ids(self):
        output = ("0, NVIDIA GeForce RTX 3090 Ti, 24564, 19, 24094\n"
                  "4, NVIDIA GeForce GTX 1060 6GB, 6144, 4000, 2066\n"
                  "5, NVIDIA GeForce GTX 1060 6GB, 6144, 4, 6062\n")
        pci = ("0, 00000000:01:00.0\n"
               "4, 00000000:11:00.0\n"
               "5, 00000000:12:00.0\n")
        with patch.object(kv_chat_control, "run", side_effect=[
                SimpleNamespace(stdout=output), SimpleNamespace(stdout=pci)]):
            cards = kv_chat_control.worker_cards(kv_chat_control.NEMO_ID)
        self.assertEqual([card["gpu"] for card in cards], [4, 5])
        self.assertEqual([card["runtime_gpu"] for card in cards], [1, 2])


class ReservationTests(unittest.TestCase):
    def test_admission_uses_driver_free_not_total_minus_used(self):
        card = {"gpu": 2, "runtime_gpu": 2, "total_mib": 8192,
                "used_mib": 5968, "free_mib": 2140}
        self.assertEqual(kv_chat_control.observed_free_mib(card), 2140)
        self.assertEqual(kv_chat_control.available_worker_cards(
            kv_chat_control.MODEL_ID, 262144, [], [card], []), [])

    def test_qwen_parent_headroom_includes_first_turn_runtime_allocation(self):
        card = {"gpu": 1, "runtime_gpu": 1, "total_mib": 6144,
                "used_mib": 78, "free_mib": 5988}
        reservations = {str(i): {"placement": {"gpus": [1]}}
                        for i in range(5)}
        self.assertEqual(kv_chat_control.available_worker_cards(
            kv_chat_control.MODEL_ID, 16384, [], [card], [], reservations,
            parent_free_mib=490), [
                {**card, "attached_chats": 5}])
        self.assertEqual(kv_chat_control.available_worker_cards(
            kv_chat_control.MODEL_ID, 16384, [], [card], [], reservations,
            parent_free_mib=489), [])

    def test_status_joins_live_occupancy_to_persisted_reservation(self):
        reservation = {"reservation_id": "r1", "context_capacity": 65536,
                       "reserved_mib": 1538, "placement": {"gpus": [1]}}
        st = {"model": kv_chat_control.MODEL_ID, "port": 11524,
              "mode": "gpu0-main-q4", "instance_id": "service-1",
              "claims": {"main": "parent", "child": "worker"},
              "reservations": {"child": reservation}}
        cards = [{"gpu": 2, "runtime_gpu": 2, "total_mib": 6144, "used_mib": 4}]
        output = StringIO()
        with patch.object(sys, "argv", ["kv_chat_control.py", "status"]), \
             patch.object(kv_chat_control, "state", return_value=st), \
             patch.object(kv_chat_control, "locked", return_value=nullcontext()), \
             patch.object(kv_chat_control, "container_running", return_value=True), \
             patch.object(kv_chat_control, "healthy", return_value=True), \
             patch.object(kv_chat_control, "request", return_value=(200, {"chats": [
                 {"chat_id": "child", "gpu": 1, "context_cap": 65536,
                  "used_tokens": 1234}]})), \
             patch.object(kv_chat_control, "worker_cards", return_value=cards), \
             patch.object(kv_chat_control, "parent_gpu_free_mib", return_value=3000), \
             patch.object(kv_chat_control, "run", return_value=SimpleNamespace(
                 stdout=json.dumps({"claims": []}))), redirect_stdout(output):
            kv_chat_control.main()
        status = json.loads(output.getvalue())
        self.assertEqual(status["chats"][0]["reservation"]["reservation_id"], "r1")
        self.assertEqual(status["chats"][0]["reservation"]["occupied_tokens"], 1234)
        self.assertEqual(status["chats"][0]["reservation"]["context_capacity"], 65536)
        self.assertEqual(status["worker_capacity"]["free_gpu_ids"], [2])

    def test_capacity_report_uses_the_attach_admission_predicate(self):
        cards = [{"gpu": 1, "total_mib": 6144, "used_mib": 10},
                 {"gpu": 2, "total_mib": 6144, "used_mib": 5000},
                 {"gpu": 3, "total_mib": 8192, "used_mib": 10}]
        chats = [{"chat_id": "main", "gpu": 0}]
        claims = [{"gpu_ids": [3]}]
        available = kv_chat_control.available_worker_cards(
            kv_chat_control.MODEL_ID, 65536, chats, cards, claims)
        self.assertEqual([card["gpu"] for card in available], [2, 1])
        self.assertEqual(kv_chat_control.available_worker_cards(
            kv_chat_control.MODEL_ID, 65536,
            chats, cards, claims, {"one": {"placement": {"gpus": [1]}},
                                   "two": {"placement": {"gpus": [2]}}}), [
                                       {**cards[1], "attached_chats": 1, "free_mib": 1144},
                                       {**cards[0], "attached_chats": 1, "free_mib": 6134}])

    def test_nemo_capacity_uses_measured_context_plus_safety_reserve(self):
        cards = [{"gpu": 5, "runtime_gpu": 5, "total_mib": 6144,
                  "used_mib": 3834}]
        self.assertEqual(len(kv_chat_control.available_worker_cards(
            kv_chat_control.NEMO_ID, 262144, [], cards, [])), 1)
        cards[0]["used_mib"] = 3835
        self.assertEqual(kv_chat_control.available_worker_cards(
            kv_chat_control.NEMO_ID, 262144, [], cards, []), [])

    def test_nemo_spreads_before_stacking_and_enforces_global_child_limit(self):
        cards = [{"gpu": 1, "runtime_gpu": 2, "total_mib": 6144, "used_mib": 4100},
                 {"gpu": 5, "runtime_gpu": 5, "total_mib": 6144, "used_mib": 2200}]
        reservations = {"old": {"placement": {"gpus": [5]}}}
        available = kv_chat_control.available_worker_cards(
            kv_chat_control.NEMO_ID, 16384, [], cards, [], reservations)
        self.assertEqual([card["gpu"] for card in available], [1, 5])
        full = {str(i): {"placement": {"gpus": [5]}}
                for i in range(kv_chat_control.NEMO_MAX_CHILDREN)}
        self.assertEqual(kv_chat_control.available_worker_cards(
            kv_chat_control.NEMO_ID, 16384, [], cards, [], full), [])

    def test_qwen_main_and_headless_counts_are_gated_by_imported_recipes(self):
        cards = [{"gpu": gpu, "runtime_gpu": gpu, "total_mib": 6144,
                  "used_mib": 10, "free_mib": 6000} for gpu in range(1, 6)]
        main_limit = kv_chat_control.max_worker_chats(
            kv_chat_control.MODEL_ID, "gpu0-main-q4")
        pool_limit = kv_chat_control.max_worker_chats(
            kv_chat_control.MODEL_ID, "gpu0-worker-pool-q4")
        self.assertEqual(main_limit, 6)
        self.assertEqual(pool_limit, 16)
        reservations = {str(i): {"placement": {"gpus": [1]}}
                        for i in range(main_limit)}
        self.assertEqual(kv_chat_control.available_worker_cards(
            kv_chat_control.MODEL_ID, 16384, [], cards, [], reservations,
            mode="gpu0-main-q4"), [])
        self.assertTrue(kv_chat_control.available_worker_cards(
            kv_chat_control.MODEL_ID, 16384, [], cards, [], reservations,
            mode="gpu0-worker-pool-q4"))
        reservations = {str(i): {"placement": {"gpus": [1]}}
                        for i in range(pool_limit)}
        self.assertEqual(kv_chat_control.available_worker_cards(
            kv_chat_control.MODEL_ID, 16384, [], cards, [], reservations,
            mode="gpu0-worker-pool-q4"), [])
        st = {"mode": "gpu0-worker-pool-q4", "reservations": {}, "claims": {}}
        with patch.object(kv_chat_control, "parent_gpu_free_mib", return_value=3000):
            capacity = kv_chat_control.capacity_report(
                kv_chat_control.MODEL_ID, 65536, [], cards, [], st)
        self.assertEqual(capacity["available_chat_slots"], pool_limit)
        self.assertEqual(capacity["max_attached_chats"], pool_limit)

    def test_qwen_reserves_full_requested_capacity_not_occupied_depth(self):
        small = kv_chat_control.estimated_worker_mib(kv_chat_control.MODEL_ID, 16384)
        full = kv_chat_control.estimated_worker_mib(kv_chat_control.MODEL_ID, 262144)
        self.assertEqual(full, 1958)
        self.assertEqual(small, 123)
        self.assertGreater(full, small)
        self.assertEqual(kv_chat_control.estimated_worker_mib(
            kv_chat_control.MODEL_ID, 262144), full)

    def test_dynamic_nemo_chat_reserves_full_requested_context(self):
        self.assertEqual(kv_chat_control.estimated_worker_mib(
            kv_chat_control.NEMO_ID, 262144, 6144), 2054)
        self.assertEqual(kv_chat_control.estimated_worker_mib(
            kv_chat_control.NEMO_ID, 16384), 134)


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
        self.assertCountEqual([c.args[0][3] for c in run.call_args_list], ["main", "worker"])

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

    def test_recovery_commits_an_attach_that_reached_the_server(self):
        reservation = {"reservation_id": "r1", "context_capacity": 32768}
        st = {"port": 11524, "claims": {"main": "parent"}, "reservations": {},
              "pending_operation": {"action": "attach", "chat_id": "child",
                                    "claim": "worker", "reservation": reservation}}
        with patch.object(kv_chat_control, "request", return_value=(
                200, {"chats": [{"chat_id": "main"}, {"chat_id": "child"}]})), \
             patch.object(kv_chat_control, "save") as save, \
             patch.object(kv_chat_control, "release") as release:
            recovered = kv_chat_control.recover_pending_operation(st)
        self.assertEqual(recovered["claims"]["child"], "worker")
        self.assertEqual(recovered["reservations"]["child"], reservation)
        self.assertNotIn("pending_operation", recovered)
        save.assert_called_once_with(recovered)
        release.assert_not_called()

    def test_recovery_rolls_back_an_attach_that_never_reached_the_server(self):
        st = {"port": 11524, "claims": {"main": "parent"}, "reservations": {},
              "pending_operation": {"action": "attach", "chat_id": "child",
                                    "claim": "worker", "reservation": {}}}
        with patch.object(kv_chat_control, "request", return_value=(
                200, {"chats": [{"chat_id": "main"}]})), \
             patch.object(kv_chat_control, "save"), \
             patch.object(kv_chat_control, "release") as release:
            recovered = kv_chat_control.recover_pending_operation(st)
        release.assert_called_once_with("worker")
        self.assertNotIn("child", recovered["claims"])
        self.assertNotIn("pending_operation", recovered)

    def test_recovery_retries_and_finishes_a_pending_detach(self):
        reservation = {"reservation_id": "r1", "context_capacity": 32768}
        st = {"port": 11524, "claims": {"main": "parent", "child": "worker"},
              "reservations": {"child": reservation},
              "pending_operation": {"action": "detach", "chat_id": "child",
                                    "claim": "worker", "reservation": reservation}}
        replies = [(200, {"chats": [{"chat_id": "main"}, {"chat_id": "child"}]}),
                   (200, {"detached": "child"})]
        with patch.object(kv_chat_control, "request", side_effect=replies) as request, \
             patch.object(kv_chat_control, "save"), \
             patch.object(kv_chat_control, "release") as release:
            recovered = kv_chat_control.recover_pending_operation(st)
        self.assertEqual(request.call_args_list[1].args,
                         (11524, "/chats/child", "DELETE"))
        release.assert_called_once_with("worker")
        self.assertNotIn("child", recovered["claims"])
        self.assertNotIn("child", recovered["reservations"])
        self.assertNotIn("pending_operation", recovered)


if __name__ == "__main__":
    unittest.main()
