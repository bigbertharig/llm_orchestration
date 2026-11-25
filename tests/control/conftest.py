import pytest
from pathlib import Path
from rig_control.manager import observations

@pytest.fixture(autouse=True)
def observation_log(tmp_path, monkeypatch):
    monkeypatch.setenv('MODEL_CATALOG', str(Path(__file__).resolve().parents[2] / 'model_catalog.json'))
    path = tmp_path / 'load_observations.jsonl'
    monkeypatch.setattr(observations, 'PATH', str(path))
    return path
