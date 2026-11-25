"""Public read-only operations: strict evidence, isolated CLI and no hidden repair."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from unittest.mock import patch

import pytest

RIG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RIG))
import inspection

CARD = {"index": 2, "name": "Example", "limit_w": 100., "default_w": 100.,
        "min_w": 50., "max_w": 150.}


@pytest.mark.parametrize("selection", ["", "-1", "2,2", "2,", "false"])
def test_invalid_selection_never_reads_hardware(selection):
    with patch.object(inspection.hw_power, "cards") as cards:
        payload, code = inspection.inspect("power", gpus=selection, config_path=Path("unused"))
    assert code == 1 and payload["kind"] == "invalid"
    cards.assert_not_called()


def test_power_success_mismatch_missing_and_unavailable_are_distinct():
    with patch.object(inspection.hw_power, "load_rules", return_value={}), \
         patch.object(inspection.hw_power, "cards", return_value=[CARD]) as cards:
        payload, code = inspection.inspect("power", gpus="2", config_path=Path("unused"))
        assert code == 0 and payload["schema"] == "rig-power-check/1"
        assert payload["checks"] == [{"gpu_id": 2, "actual_w": 100., "planned_w": 100., "matches": True}]
        cards.return_value = [{**CARD, "limit_w": 80.}]
        payload, code = inspection.inspect("power", gpus="2", config_path=Path("unused"))
        assert code == 1 and payload["kind"] == "power_mismatch"
        assert payload["checks"][0]["matches"] is False
        payload, code = inspection.inspect("power", gpus="9", config_path=Path("unused"))
        assert code == 1 and payload["kind"] == "not_found"
        cards.side_effect = FileNotFoundError("no driver")
        payload, code = inspection.inspect("power", gpus="2", config_path=Path("unused"))
        assert code == 1 and payload["kind"] == "unavailable"


@pytest.mark.parametrize("value", [float("nan"), float("inf"), 0., -1.])
def test_nonfinite_or_nonpositive_telemetry_fails(value):
    with patch.object(inspection.hw_power, "load_rules", return_value={}), \
         patch.object(inspection.hw_power, "cards", return_value=[{**CARD, "limit_w": value}]):
        payload, code = inspection.inspect("power", gpus="2", config_path=Path("unused"))
    assert code == 1 and payload["kind"] == "unavailable"


def test_model_path_uses_site_resolution_and_checks_file(tmp_path):
    hotset = tmp_path / "models" / "example" / "model.gguf"
    hotset.parent.mkdir(parents=True)
    hotset.write_bytes(b"fixture")
    config = tmp_path / "site.json"
    config.write_text(json.dumps({"model_search_roots": [str(tmp_path / "models")]}))
    requested = "/mnt/shared/models/example/model.gguf"
    payload, code = inspection.inspect("model-path", path=requested, config_path=config)
    assert code == 0 and payload["schema"] == "rig-model-path/1"
    assert payload["requested_path"] == requested and payload["resolved_path"] == str(hotset)
    hotset.unlink()
    payload, code = inspection.inspect("model-path", path=requested, config_path=config)
    assert code == 1 and payload["kind"] == "not_found"
    payload, code = inspection.inspect("model-path", path="relative.gguf", config_path=config)
    assert code == 1 and payload["kind"] == "invalid"


def test_cli_runs_without_siblings_model_artifacts_or_state_writes(tmp_path):
    # Copy only this repository's Python modules into a fresh installation, not any
    # site configuration, hardware state, model snapshot or sibling repository.
    shared = tmp_path / "install" / "shared"
    for source in [*RIG.glob("*.py"), *RIG.parent.glob("*.py"),
                   *(RIG.parents[1] / "model_library").glob("*.py")]:
        dest = shared / source.relative_to(RIG.parents[1])
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, dest)
    model = tmp_path / "model with spaces.gguf"
    model.write_bytes(b"fixture")
    home = tmp_path / "home"; home.mkdir()
    env = {**os.environ, "HOME": str(home), "RIG_SITE_CONFIG": str(tmp_path / "missing-site.json")}
    env.pop("PYTHONPATH", None)
    command = [sys.executable, str(shared / "scripts/rig/rig.py"), "inspect", "model-path", "--path", str(model)]
    result = subprocess.run(command, cwd=tmp_path, env=env, capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["resolved_path"] == str(model)
    assert not (shared / "brain").exists()
    assert not list(home.iterdir())
    model.unlink()
    result = subprocess.run(command, cwd=tmp_path, env=env, capture_output=True, text=True, timeout=20)
    assert result.returncode == 1 and json.loads(result.stdout)["kind"] == "not_found"
    assert not (shared / "brain").exists()
