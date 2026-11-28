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
        elif action == 'reset':
            row = next(r for r in self.chats if r['chat_id'] == args['chat_id'])
            assert row['used_tokens'] == args['expected_used_tokens']
            previous = row['used_tokens']
            row['used_tokens'] = 0
            return {'chat_id': args['chat_id'], 'used_tokens': 0,
                    'previous_used_tokens': previous, 'service_instance_id': self.instance}
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


def test_compact_resets_only_owned_context_without_releasing_placement(runtime):
    chat, manager, _ = runtime
    async def run():
        main = await chat.prepare('shared-35b', 'a')
        child = await chat.prepare('shared-35b', 'b')
        rows = {r['chat_id']: r for r in manager.rig.chats}
        rows['main']['used_tokens'] = 7000
        rows[child['chat_id']]['used_tokens'] = 3000
        with pytest.raises(ManagerError, match='placement changed'):
            await chat.compact('a', 'old', 'main', 7000)
        with pytest.raises(ManagerError, match='changed before compaction'):
            await chat.compact('a', 'one', 'main', 6999)
        cleared = await chat.compact('a', 'one', 'main', 7000)
        assert cleared['previous_used_tokens'] == 7000
        assert rows['main']['used_tokens'] == 0
        assert rows[child['chat_id']]['used_tokens'] == 3000
        assert (await chat.prepare('shared-35b', 'a'))['chat_id'] == main['chat_id']
        cleared_child = await chat.compact('b', 'one', child['chat_id'], 3000)
        assert cleared_child['used_tokens'] == 0
        assert len(manager.rig.chats) == 2
        assert [action for action, _ in manager.rig.calls].count('reset') == 2
    asyncio.run(run())
