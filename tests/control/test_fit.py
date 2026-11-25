"""Model fit (manager/fit.py) on synthetic GGUF headers, and `models/load` end to end against the fake rig."""

from __future__ import annotations

import json

import pytest

import gguf
from gguf import card, dense
from rig_control.manager import fit, models, observations
from test_manager import menv  # noqa: F401  (fixture: the gateway with the manager and a fake rig)

MIB = gguf.MIB


def test_shape_reads_weights_kv_and_puts_the_embedding_on_the_host(tmp_path):
    s = fit.shape_for(dense(tmp_path), "m")
    assert s.n_layers == 10 and s.n_ctx_train == 32768
    assert round(s.weights_mb) == 10 * 100 + 50  # layers + output; token_embd stays in host RAM
    assert round(s.host_bytes / MIB) == 40
    assert s.kv_per_token[0] == 2 * (128 + 128) * 34 / 32  # KV heads x (key + value) x q8_0


def test_hybrid_and_next_token_layers(tmp_path):
    path = dense(tmp_path, n_layer=9, **{"llama.full_attention_interval": 4, "llama.nextn_predict_layers": 1})
    s = fit.shape_for(path, "h")
    assert [i for i, kv in enumerate(s.kv_per_token) if kv] == [3, 7]  # every 4th; layer 8 is skipped
    assert s.weights[8] == 0 and round(s.weights_mb) == 8 * 100 + 50


def test_sliding_window_layers_cap_their_kv(tmp_path):
    path = dense(tmp_path, n_layer=4, **{"llama.attention.sliding_window": 1024,
                                         "llama.attention.sliding_window_pattern": [1, 1, 1, 0]})
    s = fit.shape_for(path, "w")
    assert s.kv_window == [1024, 1024, 1024, None]


def test_split_puts_the_output_on_the_last_card_and_balances(tmp_path):
    s = fit.shape_for(dense(tmp_path, n_layer=20, layer_mb=200, output_mb=400), "m")
    p = fit.plan(s, [card(3), card(1)], ctx_min=4096)
    assert p["fits"] and p["gpus"] == [1, 3]                     # CUDA order: lowest GPU first
    split = [int(x) for x in p["tensor_split"].split(",")]
    assert sum(split) == 20 + 1                                  # every layer plus the output slot
    a, b = p["cards"]
    assert a["layers"] + b["layers"] == 20 and abs(a["headroom_mb"] - b["headroom_mb"]) < 300
    assert min(a["headroom_mb"], b["headroom_mb"]) >= 0
    assert p["ctx"] % fit.CTX_STEP == 0 and p["ctx"] <= 32768


def test_capped_at_the_trained_context_or_the_request(tmp_path):
    s = fit.shape_for(dense(tmp_path, n_layer=4, layer_mb=50, ctx_train=8192), "small")
    assert fit.plan(s, [card(1)])["ctx"] == 8192
    assert fit.plan(s, [card(1)])["ctx_limit"] == "trained context"
    assert fit.plan(s, [card(1)], ctx=6000)["ctx"] == 5888       # exact request, rounded to the step


def test_wont_fit_says_why_and_suggests_cards(tmp_path):
    s = fit.shape_for(dense(tmp_path, n_layer=40, layer_mb=400), "big")  # 16 GB of weights
    p = fit.plan(s, [card(1), card(2)])
    assert not p["fits"] and p["reason"].startswith("won't fit: big needs about")
    alts = fit.suggest(s, [card(1), card(2), card(3), card(4), card(0, 24564)])
    assert any("GPU 0" in a for a in alts) and any(a.startswith("4 card(s)") for a in alts)


def test_partial_observations_do_not_calibrate_full_gpu_fit(tmp_path, monkeypatch):
    s = fit.shape_for(dense(tmp_path, n_layer=10, layer_mb=300, name="partial.gguf"), "partial")
    monkeypatch.setattr(observations, "PATH", str(tmp_path / "observations.jsonl"))
    observations.record("load_failed", placement="ram", source=s.source, ubatch=64,
                        oom=True, planned_ctx=8192, card_sizes=[6144])
    observations.record("load", placement="ram", source=s.source, ubatch=64,
                        n_ctx=8192, gpus=[1], cards=[{"gpu": 1, "compute_mb": 4000}])
    assert fit.failed_ceiling(s, 64, [6144]) is None
    assert fit.compute_basis(s, 64, 1).startswith("formula")


def test_ram_spill_keeps_requested_context_and_reserve(tmp_path):
    s = fit.shape_for(dense(tmp_path, n_layer=40, layer_mb=640, output_mb=200), "overflow")
    gpu = card(0, 24564)
    assert not fit.plan(s, [gpu], ctx=8192)["fits"]
    ram = fit.plan_ram(s, gpu, ctx=8192, ubatch=64, mem_available_mb=28000)
    assert ram["fits"] and ram["placement"] == "ram" and ram["ctx"] == 8192
    assert 0 < ram["n_gpu_layers"] < 40
    assert ram["host_estimate_mb"] <= ram["host_budget_mb"]
    assert ram["cards"][0]["headroom_mb"] >= ram["margin_mb"]
    assert not fit.plan_ram(s, gpu, ctx=8192, ubatch=64, mem_available_mb=4096)["fits"]


def test_harness_picks_one_card_for_small_context_and_splits_for_big(tmp_path):
    s = fit.shape_for(dense(tmp_path, n_layer=20, layer_mb=200, output_mb=400, ctx_train=262144), "m")
    free = [card(1), card(2), card(3), card(0, 24564)]
    one = fit.choose(s, free)                      # no context asked: one small card is enough
    assert one["gpus"] == [1]
    two = fit.choose(s, free, ctx=one["ctx"] * 2)  # more than one 1060 holds: split on two 1060s
    assert two["gpus"] == [1, 2]
    huge = fit.choose(s, free, ctx=one["ctx"] * 6)  # more than two 1060s hold: the big card, not three small
    assert huge["gpus"] == [0]


def test_chat_prefers_fastest_free_single_card_and_falls_back(tmp_path):
    s = fit.shape_for(dense(tmp_path, n_layer=20, layer_mb=200, output_mb=400, ctx_train=262144), "m")
    free = [card(1), card(2), card(0, 24564)]
    assert fit.choose_chat(s, free, ctx=8192)["gpus"] == [0]
    assert fit.choose_chat(s, free[:2], ctx=8192)["gpus"] == [1]
    one = fit.plan(s, [card(1)])["ctx"]
    assert fit.choose_chat(s, free[:2], ctx=one * 2)["gpus"] == [1, 2]


def test_chat_prefers_3090_overflow_before_1060_only_split(tmp_path):
    # Too large for the 3090 alone, but fits on it plus one 1060.
    s = fit.shape_for(dense(tmp_path, n_layer=40, layer_mb=640, output_mb=200), "overflow")
    free = [card(1), card(2), card(3), card(4), card(5), card(0, 24564)]
    assert not fit.plan(s, [free[-1]], ctx=8192)["fits"]
    assert fit.plan(s, [free[-1], free[0]], ctx=8192)["fits"]
    assert fit.choose_chat(s, free, ctx=8192)["gpus"] == [0, 1]
    backup = fit.choose_chat(s, free[:-1], ctx=8192)
    assert backup is not None and backup["fits"] and 0 not in backup["gpus"]


def test_chat_placement_changes_when_desired_context_grows(tmp_path):
    s = fit.shape_for(dense(tmp_path, n_layer=20, layer_mb=1000, output_mb=200,
                            ctx_train=131072, **{"llama.attention.head_count_kv": 32}), "context")
    free = [card(0, 24564), card(1), card(2)]
    assert fit.choose_chat(s, free, ctx=8192)["gpus"] == [0]
    assert not fit.plan(s, [free[0]], ctx=32768)["fits"]
    assert fit.choose_chat(s, free, ctx=32768)["gpus"] == [0, 1]


def test_measured_response_speed_can_choose_the_3090_for_a_worker(tmp_path):
    s = fit.shape_for(dense(tmp_path, n_layer=20, layer_mb=200, output_mb=400), "m")
    free = [card(1), card(2), card(0, 24564)]
    assert fit.choose(s, free, ctx=8192, ubatch=64)["gpus"] == [1]  # fallback without two measurements
    observations.record("performance", source=s.source, ubatch=64, n_ctx=8192, gpus=[1],
                        prefill_tps=400, decode_tps=20, record="one-1060")
    assert fit.choose(s, free, ctx=8192, ubatch=64)["gpus"] == [1]  # one sample cannot overrule fallback
    observations.record("performance", source=s.source, ubatch=64, n_ctx=8192, gpus=[0],
                        prefill_tps=2000, decode_tps=120, record="one-3090")
    chosen = fit.choose(s, free, ctx=8192, ubatch=64)
    assert chosen["gpus"] == [0] and chosen["performance_basis"] == "one-3090"
    assert fit.choose_smallest(s, free, ctx=8192, ubatch=64)["gpus"] == [1]
    assert fit.choose(s, free[:2], ctx=8192, ubatch=64)["gpus"] == [1]  # unavailable GPU never chosen
    assert fit.choose(s, free, ctx=8192, ubatch=128)["gpus"] == [1]  # unrelated batch retains fallback
    assert fit.choose(s, free, ctx=8192, ubatch=64, runtime_image="new-image")["gpus"] == [1]


def test_harness_prefers_the_big_card_over_many_small_ones(tmp_path):
    s = fit.shape_for(dense(tmp_path, n_layer=40, layer_mb=400), "big")  # 16 GB: needs 3+ 1060s
    p = fit.choose(s, [card(1), card(2), card(3), card(4), card(0, 24564)])
    assert p["gpus"] == [0]


def measured(n_ctx, compute_mb, gpus=(1,)):
    return {"n_ctx": n_ctx, "cards": [{"gpu": g, "model_mb": 1000, "kv_mb": 100, "compute_mb": compute_mb,
                                       "total_mb": 1100 + compute_mb} for g in gpus]}


def test_compute_buffer_is_calibrated_per_card_count(tmp_path):
    s = fit.shape_for(dense(tmp_path, ctx_train=262144), "m")
    formula = fit.compute_mb(s, 131072, 64, 2)
    assert fit.compute_basis(s, 64, 2).startswith("formula")
    assert fit.record_observation(s.source, "m", 64, [1, 2], measured(65536, 100, (1, 2)))
    split = fit.compute_mb(s, 131072, 64, 2)
    assert split < formula and fit.compute_basis(s, 64, 2) == "measured (1 load, 2 cards)"
    assert fit.compute_mb(s, 131072, 64, 1) == formula  # split evidence never calibrates a solo load
    fit.record_observation(s.source, "m", 64, [1], measured(131072, 200))
    assert abs(fit.compute_mb(s, 131072, 64, 1) - (200 * 1.1 + 64)) < 1
    assert fit.compute_mb(s, 131072, 64, 2) == split  # solo evidence never changes a split estimate
    assert fit.compute_mb(s, 131072, 512, 1) == fit._formula_mb(s, 131072, 512)


def test_measured_compute_is_never_discarded_below_the_old_formula(tmp_path):
    s = fit.shape_for(dense(tmp_path, ctx_train=262144), "m")
    fit.record_observation(s.source, "m", 64, [1], measured(8192, 5000))
    assert fit.compute_mb(s, 8192, 64) >= 5000 + fit.COMPUTE_PAD_MB
    assert not fit.record_observation(s.source, "m", 64, [1], {"n_ctx": 8192, "cards": []})
    assert not fit.record_observation(s.source, "m", 64, [1], None)


def test_calibration_lets_more_context_fit(tmp_path):
    s = fit.shape_for(dense(tmp_path, n_layer=20, layer_mb=200, output_mb=400, ctx_train=262144,
                            **{"llama.attention.head_count_kv": 8}), "m")  # KV-heavy: context is memory-bound
    before = fit.plan(s, [card(1), card(2)])
    assert before["ctx_limit"] == "card memory"
    fit.record_observation(s.source, "m", 64, [1, 2], measured(before["ctx"], 40, (1, 2)))
    after = fit.plan(s, [card(1), card(2)])
    assert after["ctx"] > before["ctx"] and after["compute_basis"] == "measured (1 load, 2 cards)"


# ------------------------------------------------------------------ end to end (fake rig)

@pytest.fixture
def registry(tmp_path, monkeypatch):
    """The gateway's own model catalog (what tools/generate_profiles.py imports), with three synthetic GGUFs."""
    small = dense(tmp_path, n_layer=20, layer_mb=200, output_mb=400, name="small.gguf")
    big = dense(tmp_path, n_layer=40, layer_mb=400, name="big.gguf")
    mid = dense(tmp_path, n_layer=20, layer_mb=400, output_mb=400, name="mid.gguf")   # needs two 1060s
    path = tmp_path / "model_catalog.json"
    gguf.write_catalog(path, {"mid:14b": ("candidate", mid), "small:9b": ("active", small), "big:30b": ("active", big)},
                       {"runtime": {"batch_size": 64, "parallel": 1}, "extra_args": ["--reasoning", "off"]})
    monkeypatch.setenv("MODEL_CATALOG", str(path))
    return path


def test_fit_endpoint_answers_without_loading(menv, registry):  # noqa: F811
    c = menv["client"]
    r = c.post("/v1/manager/models/fit", json={"model": "small:9b", "cards": 2})
    assert r.status_code == 200 and r.json()["fits"] and r.json()["gpus"] == [1, 2]
    r = c.post("/v1/manager/models/fit", json={"model": "big:30b", "gpu_ids": [1, 2]})
    assert r.status_code == 422 and "won't fit" in r.json()["error"]["message"]
    assert "GPU [0]" in r.json()["error"]["message"] or "GPU 0" in r.json()["error"]["message"]
    assert menv["leases"]() == [] and not menv["rig"]().get("plan_items")
    r = c.post("/v1/manager/models/fit", json={"model": "big:30b"})  # no cards: the harness chooses
    assert r.status_code == 200 and r.json()["gpus"] == [0]
    assert r.json()["chosen_by"] == "policy: smallest available fit"


def test_chat_auto_placement_uses_big_card_but_explicit_cards_win(menv, registry):  # noqa: F811
    c = menv["client"]
    chat = c.post("/v1/manager/models/fit", json={"model": "small:9b", "use": "chat", "ctx": 8192})
    assert chat.status_code == 200, chat.text
    assert chat.json()["gpus"] == [0]
    assert chat.json()["chosen_by"] == "policy: fastest available fit"
    assert chat.json()["placement_policy"] == "fastest_fit"
    benchmark = c.post("/v1/manager/models/fit", json={"model": "small:9b", "use": "benchmark", "ctx": 8192})
    assert benchmark.status_code == 200, benchmark.text
    assert benchmark.json()["gpus"] == [1]
    pinned = c.post("/v1/manager/models/fit",
                    json={"model": "small:9b", "use": "chat", "ctx": 8192, "gpu_ids": [1]})
    assert pinned.status_code == 200, pinned.text
    assert pinned.json()["gpus"] == [1]


def test_auto_placement_policy_can_preserve_or_prioritize_big_card(menv, registry):  # noqa: F811
    c = menv["client"]
    smallest = c.post("/v1/manager/models/fit", json={
        "model": "small:9b", "use": "chat", "ctx": 8192,
        "placement_policy": "smallest_fit"})
    assert smallest.status_code == 200, smallest.text
    assert smallest.json()["gpus"] == [1]
    assert smallest.json()["placement_policy"] == "smallest_fit"
    assert smallest.json()["chosen_by"] == "policy: smallest available fit"

    fastest = c.post("/v1/manager/models/fit", json={
        "model": "small:9b", "use": "worker", "ctx": 8192,
        "placement_policy": "fastest_fit"})
    assert fastest.status_code == 200, fastest.text
    assert fastest.json()["gpus"] == [0]
    assert fastest.json()["placement_policy"] == "fastest_fit"
    assert fastest.json()["chosen_by"] == "policy: fastest available fit"

    invalid = c.post("/v1/manager/models/fit", json={
        "model": "small:9b", "placement_policy": "random"})
    assert invalid.status_code == 400


def test_chat_ram_fallback_reaches_rig_with_explicit_partial_spec(menv, registry, tmp_path):  # noqa: F811
    cat = json.loads(registry.read_text())
    cat["models"]["big:30b"]["gguf_path"] = dense(
        tmp_path, n_layer=40, layer_mb=640, output_mb=200, name="ram.gguf")
    registry.write_text(json.dumps(cat))
    st = menv["rig"]()
    st["containers"] = [{"name": f"busy-{i}", "gpus": [i], "port": 12000 + i}
                        for i in range(1, 6)]
    st["guard"] = {"level": "ok", "mem_available_mb": 28000}
    menv["write"](st)
    c = menv["client"]
    fit_result = c.post("/v1/manager/models/fit",
                        json={"model": "big:30b", "use": "chat", "ctx": 8192})
    assert fit_result.status_code == 200, fit_result.text
    assert fit_result.json()["placement"] == "ram"
    assert fit_result.json()["gpus"] == [0]
    loaded = c.post("/v1/manager/models/load",
                    json={"model": "big:30b", "use": "chat", "ctx": 8192})
    assert loaded.status_code == 200, loaded.text
    args = menv["rig"]()["plan_items"][-1]["args"]
    spec = args["spec"]
    assert args["cache_ram"] == 0
    assert spec["allow_partial_offload"] is True
    assert 0 < spec["n_gpu_layers"] < 40
    assert spec["ctx_size"] == 8192


def test_unspecified_context_uses_the_profile_working_context(menv, registry):  # noqa: F811
    cat = json.loads(registry.read_text())
    cat["models"]["small:9b"]["profiles"]["default"]["runtime"]["ctx_size"] = 8192
    registry.write_text(json.dumps(cat))
    c = menv["client"]
    bench = c.post("/v1/manager/models/fit", json={"model": "small:9b", "use": "benchmark"})
    assert bench.status_code == 200, bench.text
    assert bench.json()["ctx"] == 8192
    chat = c.post("/v1/manager/models/fit", json={"model": "small:9b", "use": "chat"})
    assert chat.status_code == 200, chat.text
    assert chat.json()["ctx"] == 8192
    named = c.post("/v1/manager/models/fit", json={"model": "small:9b", "use": "benchmark", "ctx": 16384})
    assert named.json()["ctx"] == 16384  # a request that names a context keeps it


def test_loading_stage_caps_offered_context(menv, registry):  # noqa: F811
    catalog = json.loads(registry.read_text())
    catalog["models"]["small:9b"]["loading_stage"] = {"context_limit": 8192}
    registry.write_text(json.dumps(catalog))
    response = menv["client"].post("/v1/manager/models/fit", json={
        "model": "small:9b", "use": "chat", "ctx": 16384})
    assert response.status_code == 400
    assert "maximum 8192" in response.json()["error"]["message"]


def test_a_published_profile_without_cards_is_placed_by_the_fit_under_its_own_id(menv, registry, tmp_path):  # noqa: F811
    path = tmp_path / "remote_profiles.json"
    doc = json.loads(path.read_text())
    doc["remote_profiles"].append({"id": "chat-small", "model": "small:9b", "registry_profile": "default",
                                   "ctx_size": 8192, "use": "chat"})
    path.write_text(json.dumps(doc))
    r = menv["client"].post("/v1/manager/profiles/chat-small/load")
    assert r.status_code == 200, r.text
    item = menv["rig"]()["plan_items"][-1]
    assert item["args"]["profile"] == "chat-small" and item["args"]["spec"]["ctx_size"] >= 8192
    (lease,) = menv["leases"]()
    assert lease["controller"] == "interactive" and lease["gpu_ids"] == item["args"]["gpus"]


def test_load_model_leases_queues_the_spec_and_reports_ready(menv, registry):  # noqa: F811
    c = menv["client"]
    r = c.post("/v1/manager/models/load", json={"model": "small:9b", "gpu_ids": [1, 2],
                                                "extra_args": ["--reasoning", "on"]})
    assert r.status_code == 200, r.text
    out = r.json()
    assert out["profile"] == "small-9b-g1-2" and out["port"] == 11435 and out["state"] == "ready"
    (lease,) = menv["leases"]()
    assert lease["controller"] == "benchmarks" and lease["gpu_ids"] == [1, 2]
    item = menv["rig"]()["plan_items"][-1]
    spec = item["args"]["spec"]
    assert spec["ctx_size"] == out["ctx"] and spec["tensor_split"] == out["tensor_split"]
    assert spec["n_gpu_layers"] == "all" and spec["extra_args"] == ["--reasoning", "off", "--reasoning", "on"]
    assert item["args"]["cache_ram"] == 0  # benchmark use: no host prompt cache
    assert c.post("/v1/manager/profiles/small-9b-g1-2/unload").status_code == 200
    assert menv["leases"]() == []


def test_batch_size_override_reaches_the_fit_and_the_spec(menv, registry):  # noqa: F811
    c = menv["client"]
    f = c.post("/v1/manager/models/fit", json={"model": "small:9b", "gpu_ids": [1], "batch_size": 256})
    assert f.status_code == 200 and f.json()["ubatch"] == 256
    r = c.post("/v1/manager/models/load", json={"model": "small:9b", "gpu_ids": [1], "batch_size": 256})
    assert r.status_code == 200, r.text
    assert menv["rig"]()["plan_items"][-1]["args"]["spec"]["batch_size"] == 256


def test_failed_loads_cap_the_largest_plan_until_one_succeeds(tmp_path):
    s = fit.shape_for(dense(tmp_path, n_layer=20, layer_mb=200, output_mb=400, ctx_train=262144,
                            **{"llama.attention.head_count_kv": 8}), "m")
    free = fit.plan(s, [card(1), card(2)])["ctx"]
    observations.record("load_failed", source=s.source, ubatch=64, planned_ctx=free, oom=True, card_sizes=[6144, 6144])
    capped = fit.plan(s, [card(1), card(2)])
    assert capped["ctx"] < free and capped["failed_ceiling"] == free
    assert "failed_ceiling" not in fit.plan(s, [card(1)])  # another card set is not capped by it
    fit.record_observation(s.source, "m", 64, [1], measured(free * 2, 100, (1,)), hw=[{"index": 1, "total_mb": 6144}])
    assert fit.plan(s, [card(1), card(2)])["failed_ceiling"] == free  # a success on other cards does not lift it
    fit.record_observation(s.source, "m", 64, [1, 2], measured(free, 100, (1, 2)),
                           hw=[{"index": 1, "total_mb": 6144}, {"index": 2, "total_mb": 6144}])  # loaded after all
    assert "failed_ceiling" not in fit.plan(s, [card(1), card(2)])


def test_overhead_per_card_size_is_learned_from_memory_in_use(tmp_path):
    s = fit.shape_for(dense(tmp_path, ctx_train=262144), "m")
    assert fit.overhead_mb(24564) == fit.CUDA_CONTEXT_MB
    fit.record_observation(s.source, "m", 64, [0], measured(8192, 100, (0,)), used_mb=[1700],
                           hw=[{"index": 0, "total_mb": 24564}])  # 1700 in use, llama.cpp counted 1200
    assert abs(fit.overhead_mb(24564) - (500 + 64)) < 1
    assert fit.overhead_mb(6144) == fit.CUDA_CONTEXT_MB  # another card size: not measured yet


def test_depth_margin_is_limited_to_the_measured_model_and_layout(tmp_path):
    s = fit.shape_for(dense(tmp_path), "m")
    one = [card(0, 24564)]
    two = [card(1), card(2)]
    assert fit.growth_margin_mb(s, one) == fit.MARGIN_MB_FALLBACK
    observations.record("depth_memory", source=s.source, card_sizes=[24564], growth_mb=[18],
                        n_ctx=195840, model="m")
    assert fit.growth_margin_mb(s, one) == 128
    assert fit.growth_margin_mb(s, two) == fit.MARGIN_MB_FALLBACK
    assert fit.plan(s, one, ctx=8192)["margin_mb"] == 128
    assert fit.plan(s, one, ctx=8192, margin_mb=256)["margin_mb"] == 256


def test_recorded_oom_ceiling_blocks_explicit_context_too(tmp_path):
    s = fit.shape_for(dense(tmp_path, n_layer=20, layer_mb=200, output_mb=400, ctx_train=262144), "m")
    observations.record("load_failed", source=s.source, ubatch=64, planned_ctx=120064,
                        oom=True, card_sizes=[24564])
    p = fit.plan(s, [card(0, 24564)], ctx=120064)
    assert not p["fits"] and p["failed_ceiling"] == 120064
    assert "previously ran out of memory" in p["reason"]
    assert fit.plan(s, [card(0, 24564)], ctx=119808)["fits"]


def test_observations_list_the_log_by_kind(menv):  # noqa: F811
    observations.record("load_failed", model="m:1b", planned_ctx=4096, oom=True)
    c = menv["client"]
    got = c.get("/v1/manager/observations", params={"kind": "load_failed"}).json()
    assert got["count"] == 1 and got["rows"][0]["model"] == "m:1b"


def test_cards_from_the_scan_carry_uuid_and_link():
    obs = {"gpu_scan": {"cards": [{"index": 0, "uuid": "GPU-a", "name": "3090", "total_mb": 24564, "bus": "01:00.0",
                                   "link_gen": 2, "link_width": 16, "root_port": "00:01.0", "via": []},
                                  {"index": 3, "uuid": "GPU-b", "link_gen": 1, "link_width": 1, "via": ["04:00.0"]}]}}
    hw = observations.cards_from(obs, [3, 0, 7])
    assert [h["uuid"] for h in hw] == ["GPU-b", "GPU-a"] and hw[0]["via"] == ["04:00.0"]
    assert observations.cards_from({}, [0]) == []


def test_load_model_that_wont_fit_never_reaches_the_rig(menv, registry):  # noqa: F811
    r = menv["client"].post("/v1/manager/models/load", json={"model": "big:30b", "gpu_ids": [1, 2]})
    assert r.status_code == 422
    assert menv["leases"]() == [] and not menv["rig"]().get("plan_items")


# ------------------------------------------------------------------ reservations and overrides (MANAGER.md)

def _hold(c, gpu, owner="lanes"):
    r = c.post("/v1/manager/leases", json={"controller": "benchmarks", "owner": owner, "run_id": f"hold-{gpu}",
                                           "gpu_ids": [gpu], "model": "x"})
    assert r.status_code == 201, r.text
    return r.json()["lease"]["lease_id"]


def test_split_load_reserves_busy_cards_and_loads_once_they_free(menv, registry, monkeypatch):  # noqa: F811
    import threading
    import time

    from rig_control.manager import core
    monkeypatch.setattr(core, "WAIT_POLL_S", 0.1)
    c = menv["client"]
    held = _hold(c, 1)
    out = {}
    t = threading.Thread(target=lambda: out.update(r=c.post("/v1/manager/models/load", json={
        "model": "small:9b", "gpu_ids": [1, 2], "wait_s": 20})))
    t.start()
    deadline = time.time() + 10
    while time.time() < deadline and not menv["rig"]()["claims"]:
        time.sleep(0.05)
    (claim,) = menv["rig"]()["claims"]
    assert claim["name"] == "wait-small-9b-g1-2" and claim["gpu_ids"] == [1, 2] and claim["expires_at"]
    # The free partner card cannot be taken by single work while the split waits for GPU 1.
    single = c.post("/v1/manager/leases", json={"controller": "benchmarks", "owner": "solo", "run_id": "s",
                                                "gpu_ids": [2]})
    assert single.status_code == 409 and "wait-small-9b-g1-2" in single.json()["error"]["message"]
    assert c.post(f"/v1/manager/leases/{held}/release", json={"owner": "lanes"}).status_code == 200
    t.join(timeout=20)
    assert out["r"].status_code == 200, out["r"].text
    assert out["r"].json()["gpus"] == [1, 2]
    assert menv["rig"]()["claims"] == []          # the lease holds the cards now; the reservation is gone
    assert [l["gpu_ids"] for l in menv["leases"]()] == [[1, 2]]


def test_a_wait_that_runs_out_drops_its_reservation(menv, registry, monkeypatch):  # noqa: F811
    from rig_control.manager import core
    monkeypatch.setattr(core, "WAIT_POLL_S", 0.1)
    c = menv["client"]
    _hold(c, 1)
    r = c.post("/v1/manager/models/load", json={"model": "small:9b", "gpu_ids": [1, 2], "wait_s": 0.4})
    assert r.status_code == 504 and "GPU 1 leased by benchmarks/lanes" in r.json()["error"]["message"]
    assert menv["rig"]()["claims"] == [] and not menv["rig"]().get("plan_items")


def _wait_for(pred, timeout=10):
    import time
    deadline = time.time() + timeout
    while time.time() < deadline:
        if pred():
            return True
        time.sleep(0.05)
    return False


def test_load_naming_no_cards_collects_cards_as_they_free(menv, registry, monkeypatch):  # noqa: F811
    import threading

    from rig_control.manager import core
    monkeypatch.setattr(core, "WAIT_POLL_S", 0.1)
    c = menv["client"]
    held = _hold(c, 1)                       # mid:14b needs both 1060s; GPU 1 is busy, GPU 2 and the 3090 free
    st = menv["rig"]()
    st["gpus"][1]["used_mb"] = 5000          # the busy card shows its model's memory, as on the rig
    menv["write"](st)
    out = {}
    t = threading.Thread(target=lambda: out.update(r=c.post("/v1/manager/models/load", json={
        "model": "mid:14b", "wait_s": -1})))
    t.start()
    assert _wait_for(lambda: menv["rig"]()["claims"])
    (claim,) = menv["rig"]()["claims"]
    assert claim["name"].startswith("wait-mid-14b-") and claim["gpu_ids"] == [2]   # the free 1060, not the 3090
    single = c.post("/v1/manager/leases", json={"controller": "benchmarks", "owner": "solo", "run_id": "s",
                                                "gpu_ids": [2]})
    assert single.status_code == 409                                                # held for the waiting load
    assert c.post(f"/v1/manager/leases/{held}/release", json={"owner": "lanes"}).status_code == 200
    st = menv["rig"]()
    st["gpus"][1]["used_mb"] = 4             # its model unloaded with the lease
    menv["write"](st)
    t.join(timeout=20)
    assert out["r"].status_code == 200, out["r"].text
    assert out["r"].json()["gpus"] == [1, 2]
    assert menv["rig"]()["extends"] == [{"name": claim["name"], "gpus": [1]}]       # grown, never re-made
    assert menv["rig"]()["claims"] == []                                            # the lease holds them now
    assert [l["gpu_ids"] for l in menv["leases"]()] == [[1, 2]]


def test_removing_the_claim_cancels_a_collecting_wait(menv, registry, monkeypatch):  # noqa: F811
    import threading

    from rig_control.manager import core
    monkeypatch.setattr(core, "WAIT_POLL_S", 0.1)
    c = menv["client"]
    _hold(c, 1)
    out = {}
    t = threading.Thread(target=lambda: out.update(r=c.post("/v1/manager/models/load", json={
        "model": "mid:14b", "wait_s": -1})))
    t.start()
    assert _wait_for(lambda: menv["rig"]()["claims"])
    name = menv["rig"]()["claims"][0]["name"]
    assert c.post(f"/v1/manager/claims/{name}/release").status_code == 200
    t.join(timeout=20)
    assert out["r"].status_code == 409 and "cancelled" in out["r"].json()["error"]["message"]
    assert menv["rig"]()["claims"] == [] and not menv["rig"]().get("plan_items")


def test_removing_the_reservation_cancels_an_unlimited_split_wait(menv, registry, monkeypatch):  # noqa: F811
    import threading

    from rig_control.manager import core
    monkeypatch.setattr(core, "WAIT_POLL_S", 0.1)
    c = menv["client"]
    _hold(c, 1)
    out = {}
    t = threading.Thread(target=lambda: out.update(r=c.post("/v1/manager/models/load", json={
        "model": "small:9b", "gpu_ids": [1, 2], "wait_s": -1})))
    t.start()
    assert _wait_for(lambda: menv["rig"]()["claims"])
    assert c.post("/v1/manager/claims/wait-small-9b-g1-2/release").status_code == 200
    t.join(timeout=20)
    assert out["r"].status_code == 409 and "cancelled" in out["r"].json()["error"]["message"]
    assert not menv["rig"]().get("plan_items")


def test_without_wait_a_busy_card_still_fails_at_once(menv, registry):  # noqa: F811
    c = menv["client"]
    _hold(c, 1)
    r = c.post("/v1/manager/models/load", json={"model": "small:9b", "gpu_ids": [1, 2]})
    assert r.status_code == 409 and menv["rig"]()["claims"] == []


def test_kill_stops_runtimes_releases_leases_and_drops_claims(menv):  # noqa: F811
    c = menv["client"]
    lid = _hold(c, 2)
    st = menv["rig"]()
    st["containers"] = [{"name": "llama-lane-gpu2", "gpus": [2], "port": 11436}]
    menv["write"](st)
    assert c.post("/v1/manager/claims", json={"name": "pair", "gpu_ids": [1, 2], "owner": "lane-1-2",
                                              "ttl_s": 600}).status_code == 201
    r = c.post("/v1/manager/kill", json={"gpu_ids": [2], "reason": "test"})
    assert r.status_code == 200, r.text
    out = r.json()
    assert out["stopped"] == ["llama-lane-gpu2"] and out["released"] == [lid] and out["claims_dropped"] == ["pair"]
    st = menv["rig"]()
    assert st["leases"] == [] and st["claims"] == [] and st["containers"] == []
    events = c.get("/v1/manager/status").json()["rig"]["events"]
    assert any("kill GPU [2] (test)" in e["event"] for e in events)


def test_force_load_kills_what_runs_on_the_cards_first(menv, registry):  # noqa: F811
    c = menv["client"]
    _hold(c, 1, owner="someone-else")
    r = c.post("/v1/manager/models/load", json={"model": "small:9b", "gpu_ids": [1, 2], "force": True})
    assert r.status_code == 200, r.text
    assert [l["owner"] for l in menv["leases"]()] == ["orchestration"]


def test_claims_carry_owner_and_ttl_and_renew(menv):  # noqa: F811
    c = menv["client"]
    r = c.post("/v1/manager/claims", json={"name": "lane-4-5", "gpu_ids": [1, 2], "owner": "lane", "ttl_s": 600,
                                           "purpose": "benchmark lane"})
    assert r.status_code == 201 and r.json()["claim"]["expires_at"]
    assert c.post("/v1/manager/claims/lane-4-5/renew", json={"owner": "lane", "ttl_s": 600}).status_code == 200
    row = {g["gpu"]: g for g in c.get("/v1/manager/status").json()["gpus"]}[1]
    assert row["claim"] == "lane-4-5" and row["claim_info"]["owner"] == "lane"
    assert c.post("/v1/manager/claims/nope/renew", json={}).status_code == 404


# ------------------------------------------------------------------ runs, cancel by owner, outages, demand

def test_a_run_is_a_job_in_the_report_and_cancel_owner_cleans_up_what_it_holds(menv):
    c = menv["client"]
    r = c.post("/v1/manager/runs", json={"run": "camp-1", "argv": ["python3", "run_campaign.py", "m.json"],
                                          "resume_argv": ["python3", "run_campaign.py", "m.json", "--run-id", "camp-1"]})
    assert r.status_code == 201, r.text
    (run,) = c.get("/v1/manager/runs").json()["runs"]
    assert run["run"] == "camp-1" and run["state"] == "running" and run["owner"] == "camp-1"
    assert c.post("/v1/manager/runs", json={"run": "camp-1", "argv": ["x"]}).status_code == 409   # not twice
    lease = c.post("/v1/manager/leases", json={"controller": "benchmarks", "owner": "campaign-gpu_1", "run_id": "camp-1",
                                                "gpu_ids": [1]}).json()["lease"]["lease_id"]
    assert c.post("/v1/manager/claims", json={"name": "campaign-camp-1-gpu_2", "gpu_ids": [2],
                                              "owner": "campaign-camp-1"}).status_code == 201
    out = c.post("/v1/manager/owners/camp-1/cancel", json={"reason": "test"}).json()
    assert out["runs_stopped"] == ["camp-1"] and out["leases_released"] == [lease]
    assert out["claims_dropped"] == ["campaign-camp-1-gpu_2"] and not out["errors"]
    st = menv["rig"]()
    assert st["leases"] == [] and st["claims"] == [] and st["runs"][0]["state"] == "stopped"
    events = c.get("/v1/manager/status").json()["rig"]["events"]
    assert any(e["event"].startswith("cancel camp-1 (test)") for e in events)


def test_an_interrupted_run_is_offered_then_resumed_from_its_checkpoint(menv):
    c = menv["client"]
    resume = ["python3", "run_campaign.py", "m.json", "--run-id", "camp-2"]
    c.post("/v1/manager/runs", json={"run": "camp-2", "argv": ["python3", "run_campaign.py", "m.json"],
                                     "resume_argv": resume})
    st = menv["rig"]()
    st["runs"][0].update(state="interrupted", ended_at="2026-10-22T20:00:00Z", result="interrupted: the rig restarted")
    menv["write"](st)
    assert c.post("/v1/manager/reconcile").status_code == 200
    events = c.get("/v1/manager/status").json()["rig"]["events"]
    assert any("rigctl recover camp-2" in e["event"] for e in events)
    assert menv["rig"]()["runs"][0]["state"] == "interrupted"            # offered, not restarted on its own
    r = c.post("/v1/manager/runs/camp-2/recover")
    assert r.status_code == 200, r.text
    (run,) = menv["rig"]()["runs"]
    assert run["state"] == "running" and run["argv"] == resume and run["attempts"] == 2
    assert c.post("/v1/manager/runs/camp-2/recover").status_code == 409  # running: nothing to resume


def test_a_run_that_opted_in_resumes_on_its_own(menv):
    c = menv["client"]
    c.post("/v1/manager/runs", json={"run": "night", "argv": ["a"], "resume_argv": ["a", "--resume"], "auto_resume": True})
    st = menv["rig"]()
    st["runs"][0].update(state="interrupted", ended_at="2026-10-22T20:00:00Z")
    menv["write"](st)
    c.post("/v1/manager/reconcile")
    (run,) = menv["rig"]()["runs"]
    assert run["state"] == "running" and run["argv"] == ["a", "--resume"]


def test_dismissing_an_ended_run_cleans_up_and_a_running_one_is_refused(menv):
    c = menv["client"]
    c.post("/v1/manager/runs", json={"run": "lane-x", "argv": ["a"]})
    assert c.post("/v1/manager/runs/lane-x/dismiss").status_code == 409
    st = menv["rig"]()
    st["runs"][0]["state"] = "failed"
    menv["write"](st)
    out = c.post("/v1/manager/runs/lane-x/dismiss").json()
    assert out["run"]["state"] == "dismissed" and out["cleanup"]["owner"] == "lane-x"


def test_a_small_model_does_not_take_a_card_a_run_is_waiting_for(menv, registry):  # noqa: F811
    c = menv["client"]
    _hold(c, 1)
    _hold(c, 2)                              # both 1060s busy: small:9b would spill onto the 3090
    c.post("/v1/manager/runs", json={"run": "brain-q", "argv": ["a"], "needs": [{"card_mb": 24564, "count": 1}]})
    r = c.post("/v1/manager/models/load", json={"model": "small:9b"})
    assert r.status_code == 409 and "kept free" in r.json()["error"]["message"]
    r = c.post("/v1/manager/models/load", json={"model": "small:9b", "owner": "brain-q"})   # its own need
    assert r.status_code == 200, r.text
    assert r.json()["gpus"] == [0]


def test_a_collecting_wait_rides_out_an_unreachable_rig(menv, registry, monkeypatch):  # noqa: F811
    import os
    import threading

    from rig_control.manager import core
    monkeypatch.setattr(core, "WAIT_POLL_S", 0.1)
    c = menv["client"]
    held = _hold(c, 1)
    out = {}
    t = threading.Thread(target=lambda: out.update(r=c.post("/v1/manager/models/load", json={
        "model": "mid:14b", "wait_s": -1, "owner": "lane-z"})))
    t.start()
    assert _wait_for(lambda: menv["rig"]()["claims"])
    down = menv["state"] + ".down"
    open(down, "w").close()                  # the rig drops off the network
    import time
    time.sleep(0.6)
    os.remove(down)                          # and comes back
    assert c.post(f"/v1/manager/leases/{held}/release", json={"owner": "lanes"}).status_code == 200
    t.join(timeout=20)
    assert out["r"].status_code == 200, out["r"].text
    events = [e["event"] for e in c.get("/v1/manager/status").json()["rig"]["events"]]
    assert any("rig unreachable; still waiting" in e for e in events)
    assert any("rig back after" in e for e in events)
    (lease,) = [l for l in menv["leases"]() if l["gpu_ids"] == [1, 2]]
    assert lease["metadata"]["requested_by"] == "lane-z"


def test_cancel_owner_stops_its_collecting_wait(menv, registry, monkeypatch):  # noqa: F811
    import threading

    from rig_control.manager import core
    monkeypatch.setattr(core, "WAIT_POLL_S", 0.1)
    c = menv["client"]
    _hold(c, 1)
    out = {}
    t = threading.Thread(target=lambda: out.update(r=c.post("/v1/manager/models/load", json={
        "model": "mid:14b", "wait_s": -1, "owner": "lane-q"})))
    t.start()
    assert _wait_for(lambda: menv["rig"]()["claims"])
    cleanup = c.post("/v1/manager/owners/lane-q/cancel").json()
    assert cleanup["claims_dropped"] and not cleanup["errors"]
    t.join(timeout=20)
    assert out["r"].status_code == 409 and "cancelled" in out["r"].json()["error"]["message"]


def test_profile_context_modes_and_trained_limit(menv, registry):  # noqa: F811
    cat = json.loads(registry.read_text())
    p = cat["models"]["small:9b"]["profiles"]["default"]
    p["runtime"]["ctx_size"] = 16384
    p["context_bounds"] = {"min": 8192, "max": 32768, "max_basis": "gguf_train"}
    registry.write_text(json.dumps(cat))
    c = menv["client"]

    working = c.post("/v1/manager/models/fit",
                     json={"model": "small:9b", "gpu_ids": [1]})
    minimum = c.post("/v1/manager/models/fit",
                     json={"model": "small:9b", "gpu_ids": [1], "context_mode": "min"})
    maximum = c.post("/v1/manager/models/fit",
                     json={"model": "small:9b", "gpu_ids": [1, 2], "context_mode": "max"})
    assert working.status_code == minimum.status_code == maximum.status_code == 200
    assert working.json()["ctx"] == 16384
    assert minimum.json()["ctx"] == 8192
    assert maximum.json()["ctx"] > 16384
    assert maximum.json()["ctx"] <= 32768
    assert maximum.json()["context_bounds"]["max_basis"] == "gguf_train"

    auto_max = c.post("/v1/manager/models/fit",
                      json={"model": "small:9b", "context_mode": "max"})
    assert auto_max.status_code == 200 and auto_max.json()["gpus"] == [0, 1, 2]
    above = c.post("/v1/manager/models/fit",
                   json={"model": "small:9b", "ctx": 65536})
    assert above.status_code == 400 and "maximum" in above.json()["error"]["message"]
    ambiguous = c.post("/v1/manager/models/fit",
                       json={"model": "small:9b", "ctx": 8192, "context_mode": "min"})
    assert ambiguous.status_code == 400


def test_smaller_same_card_context_is_suggested(menv, registry):  # noqa: F811
    cat = json.loads(registry.read_text())
    p = cat["models"]["small:9b"]["profiles"]["default"]
    p["runtime"]["ctx_size"] = 32768
    p["context_bounds"] = {"min": 8192, "max": 32768}
    registry.write_text(json.dumps(cat))
    c = menv["client"]
    working = c.post("/v1/manager/models/fit",
                     json={"model": "small:9b", "gpu_ids": [1], "margin_mb": 1370})
    minimum = c.post("/v1/manager/models/fit",
                     json={"model": "small:9b", "gpu_ids": [1], "context_mode": "min", "margin_mb": 1370})
    assert working.status_code == 422 and minimum.status_code == 200
    assert any("context_mode=min" in x for x in working.json()["error"]["suggestions"])


def test_card_count_searches_other_available_card_classes(menv, registry):  # noqa: F811
    c = menv["client"]
    p = c.post("/v1/manager/models/fit", json={"model": "big:30b", "cards": 2})
    assert p.status_code == 200 and p.json()["gpus"] == [0, 1]
    p = c.post("/v1/manager/models/fit", json={"model": "big:30b", "cards": 1})
    assert p.status_code == 200 and p.json()["gpus"] == [0]
