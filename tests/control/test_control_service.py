import asyncio
import json
import sqlite3
from pathlib import Path
import pytest
from fastapi.testclient import TestClient
from rig_control import service

class Manager:
    calls = []
    fail = False
    def __init__(self, rig, *args, **kwargs):
        self.rig = rig
    async def load_model(self, model):
        self.calls.append(model)
        if self.fail:
            raise RuntimeError('lost worker after dispatch')
        return {'model': model, 'owner_decision': True}
    async def _require_control(self):
        pass
    async def status(self):
        return {'authority': 'orchestration'}

class Rig:
    def __init__(self):
        self.rows = []
    async def status(self):
        return {'profiles': self.rows}

def envelope(op='manager.load_model', args=None, rid='one'):
    return {'schema': service.SCHEMA, 'request_id': rid, 'operation': op,
            'arguments': args if args is not None else {'args': ['local-model'], 'kwargs': {}}}

@pytest.fixture
def control(tmp_path, monkeypatch):
    monkeypatch.setattr(service, 'Manager', Manager)
    Manager.calls = []
    Manager.fail = False
    profile = tmp_path / 'profiles.json'
    profile.write_text(json.dumps({'remote_profiles': [{'id': 'local', 'model': 'local-model', 'ctx_size': 8192}]}))
    rig = Rig()
    app = service.create_app(token='fixture', state_root=tmp_path, profiles_path=profile, rig=rig, reconcile=False)
    with TestClient(app) as c:
        c.headers['Authorization'] = 'Bearer fixture'
        yield c, rig, tmp_path

def test_auth_version_and_unknown_operation_are_refused(control):
    c, _, _ = control
    assert c.post('/v1/control', headers={'Authorization': 'Bearer wrong'}, json=envelope()).status_code == 401
    assert c.post('/v1/control', json={**envelope(), 'schema': 'orchestration-control/2'}).status_code == 400
    result = c.post('/v1/control', json=envelope('shell.execute', {})).json()
    assert result['ok'] is False and result['error']['status'] == 400
    assert Manager.calls == []

def test_duplicate_write_is_stable_and_collision_refused(control):
    c, _, _ = control
    first = c.post('/v1/control', json=envelope()).json()
    assert first['result']['owner_decision']
    assert c.post('/v1/control', json=envelope()).json() == first
    collision = c.post('/v1/control', json=envelope(args={'args': ['other'], 'kwargs': {}})).json()
    assert collision['error']['kind'] == 'id_collision'
    assert Manager.calls == ['local-model']

def test_uncertain_dispatch_is_never_replayed_after_service_restart(control, monkeypatch):
    c, rig, root = control
    Manager.fail = True
    assert c.post('/v1/control', json=envelope()).json()['error']['kind'] == 'reconciliation_required'
    # A new app over the same store sees the original durable dispatch marker.
    other = service.create_app(token='fixture', state_root=root, rig=rig, reconcile=False)
    request = TestClient(other)
    response = request.post('/v1/control', headers={'Authorization': 'Bearer fixture'}, json=envelope()).json()
    assert response['error']['kind'] == 'reconciliation_required'
    assert Manager.calls == ['local-model']

def test_ready_only_never_dispatches_load(control):
    c, rig, _ = control
    request = envelope('route.prepare', {'profile': 'local', 'ready_only': True})
    assert c.post('/v1/control', json=request).json()['error']['kind'] == 'model_not_ready'
    rig.rows = [{'id': 'local', 'model': 'local-model', 'state': 'ready', 'port': 11434, 'ctx_size': 8192}]
    assert c.post('/v1/control', json=request).json()['result'] == {'port': 11434}
    rig.rows[0]['model'] = 'another-model'
    assert c.post('/v1/control', json=request).json()['error']['kind'] == 'model_not_ready'
    assert Manager.calls == []

def test_missing_optional_profile_catalog_still_exposes_cli_service(tmp_path):
    app = service.create_app(token='fixture', state_root=tmp_path, rig=Rig(), reconcile=False)
    with TestClient(app) as c:
        response = c.post('/v1/control', headers={'Authorization': 'Bearer fixture'}, json=envelope('capabilities', {})).json()
        assert response['result']['local_runtime_authority'] is True
