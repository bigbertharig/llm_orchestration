import asyncio
import sqlite3
import pytest
from rig_control.chat_runtime import ChatRuntime
from rig_control.manager import models
from rig_control.manager.errors import ManagerError

class Rig:
    def __init__(self):
        self.instance = 'one'
        self.model = 'qwen3.6:35b-a3b-xl'
        self.chats = []
        self.calls = []
    async def status(self):
        return {'shared_kv_service': {'status': 'ready' if self.chats else 'stopped',
                'model': self.model, 'instance_id': self.instance, 'chats': self.chats}}
    async def shared_kv(self, action, **args):
        self.calls.append((action, args))
        if action == 'start':
            self.model = args['model']
            self.chats = [{'chat_id': 'main', 'gpu': 0, 'context_cap': 8192, 'used_tokens': 0}]
        elif action == 'attach':
            self.chats.append({'chat_id': args['chat_id'], 'gpu': 1, 'context_cap': 4096, 'used_tokens': 0})
        elif action == 'detach':
            self.chats = [r for r in self.chats if r['chat_id'] != args['chat_id']]
        elif action == 'stop':
            self.chats = []

class Manager:
    def __init__(self):
        self.rig = Rig()
        self.controls = 0
    async def _require_control(self):
        self.controls += 1
    async def chat_availability(self):
        return {'ready_profiles': []}

@pytest.fixture
def runtime(tmp_path, monkeypatch):
    monkeypatch.setattr(models, 'catalog', lambda: {'models': {'qwen3.6:35b-a3b-xl': {'status': 'active'}}})
    manager = Manager()
    db = lambda: sqlite3.connect(tmp_path / 'state.sqlite3')
    return ChatRuntime(manager, lambda: {}, db), manager, db

def test_owner_assigns_main_once_and_persists_binding_across_restart(runtime):
    chat, manager, db = runtime
    async def run():
        first = await chat.prepare('shared-35b', 'conversation-a')
        second = await chat.prepare('shared-35b', 'conversation-b')
        assert first['chat_id'] == 'main' and second['gpu_ids'] == [1]
        restarted = ChatRuntime(manager, lambda: {}, db)
        assert (await restarted.prepare('shared-35b', 'conversation-b'))['chat_id'] == second['chat_id']
        assert [a for a, _ in manager.rig.calls] == ['start', 'attach']
        assert manager.controls == 2
    asyncio.run(run())

def test_stale_unload_and_main_park_are_owner_refusals(runtime):
    chat, manager, _ = runtime
    async def run():
        await chat.prepare('shared-35b', 'a')
        with pytest.raises(ManagerError, match='Parking requires'):
            await chat.release('a', park=True)
        with pytest.raises(ManagerError, match='pinned'):
            await chat.release('a', 'one', 'main')
        with pytest.raises(ManagerError, match='instance changed'):
            await chat.reset('shared-35b', ['shared-35b'], unload=True, expected_instance='old')
        assert manager.rig.chats
        result = await chat.reset('shared-35b', ['shared-35b'], unload=True, expected_instance='one')
        assert result['remaining'] == [] and manager.rig.chats == []
    asyncio.run(run())

def test_stale_offer_cannot_select_unapproved_runtime(runtime):
    chat, manager, _ = runtime
    with pytest.raises(ManagerError, match='Unknown local chat offer'):
        asyncio.run(chat.prepare('unreviewed', 'a'))
    with pytest.raises(ManagerError, match='Unknown local chat offer'):
        asyncio.run(chat.switch('unreviewed', 'a'))
    assert manager.rig.calls == []
