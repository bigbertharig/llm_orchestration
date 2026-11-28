"""Owner-side chat offers and local model transitions."""
import asyncio
import json
from .manager import models
from .manager.errors import ManagerError
from .rig import RigUnavailable
from .shared_chat_contract import serving_approved

SHARED_NEMO_MODEL = "nemotron-3.5-bolt:74b-a9b"

SHARED_NEMO_PROFILE = "shared-nemo"

SHARED_35B_MODEL = "qwen3.6:35b-a3b-xl"

SHARED_35B_PROFILE = "shared-35b"

SHARED_PROFILES = {
    SHARED_NEMO_PROFILE: {"model": SHARED_NEMO_MODEL, "prefix": "shared-nemo-",
        "load_estimate_seconds": 150,
        "description": "Shared-weight Nemotron: one main chat plus attached GPU chats."},
    SHARED_35B_PROFILE: {"model": SHARED_35B_MODEL, "prefix": "shared-35b-",
        "load_estimate_seconds": 120,
        "description": "Shared-weight 35B: one main chat plus attached GPU chats."},
}

class ChatRuntime:
    def __init__(self, manager, profiles, db):
        self.manager, self.profiles, self.db = manager, profiles, db
        self.lock = asyncio.Lock()
        with self.db() as con:
            con.execute("CREATE TABLE IF NOT EXISTS chat_bindings (owner TEXT PRIMARY KEY, model TEXT, instance TEXT, chat TEXT)")

    def offers(self):
        choices = {pid: spec for pid, spec in self.profiles().items()
                   if spec.get('dashboard_chat', 'chatbot' in spec.get('good_for', []))
                   and spec.get('status') == 'active' and spec.get('use') == 'chat'}
        try:
            catalog = models.catalog()
            for pid, spec in SHARED_PROFILES.items():
                entry = catalog.get('models', {}).get(spec['model'], {})
                if serving_approved(catalog, spec['model']) and entry.get('status') == 'active' and not entry.get('in_testing'):
                    choices[pid] = {**spec, 'status': 'active', 'use': 'chat', 'dashboard_chat': True}
        except (OSError, ValueError, KeyError):
            pass
        return choices

    async def loaded(self):
        available = await self.manager.chat_availability()
        service = (await self.manager.rig.status()).get('shared_kv_service') or {}
        shared = next((pid for pid, spec in SHARED_PROFILES.items()
                       if spec['model'] == service.get('model') and service.get('status') == 'ready'), None)
        profiles = available.get('ready_profiles', [])
        return profiles + ([shared] if shared else []), profiles, shared, service

    async def prepare(self, profile, owner):
        if profile not in self.offers():
            raise ManagerError(404, 'Unknown local chat offer')
        if profile not in SHARED_PROFILES:
            return await self.manager.prepare_chat_profile(profile, self.profiles()[profile])
        if not isinstance(owner, str) or not owner or len(owner) > 128:
            raise ManagerError(400, 'A bounded chat owner identity is required')
        async with self.lock:
            spec = SHARED_PROFILES[profile]
            service = (await self.manager.rig.status()).get('shared_kv_service') or {}
            if service.get('status') != 'ready' or service.get('model') != spec['model']:
                if service.get('status') == 'ready':
                    raise ManagerError(409, 'Another shared model is loaded; switch models explicitly')
                await self.manager._require_control()
                await self.manager.rig.shared_kv('start', model=spec['model'])
                service = (await self.manager.rig.status()).get('shared_kv_service') or {}
            instance = service.get('instance_id')
            if service.get('status') != 'ready' or service.get('model') != spec['model'] or not instance:
                raise ManagerError(503, 'Shared service has no ready instance')
            chats = {c['chat_id']: c for c in service.get('chats', [])}
            with self.db() as con:
                binding = con.execute('SELECT instance, chat FROM chat_bindings WHERE owner=?', (owner,)).fetchone()
                claimed = con.execute('SELECT 1 FROM chat_bindings WHERE instance=? AND chat=?', (instance, 'main')).fetchone()
            chat = binding[1] if binding and binding[0] == instance and binding[1] in chats else None
            if chat is None:
                if 'main' in chats and chats['main'].get('used_tokens') == 0 and not claimed:
                    chat = 'main'
                else:
                    import hashlib
                    chat = 'dash-' + hashlib.sha256(owner.encode()).hexdigest()[:24]
                    await self.manager._require_control()
                    try:
                        await self.manager.rig.shared_kv('attach', model=spec['model'], chat_id=chat)
                    except RigUnavailable as exc:
                        if any(reason in str(exc) for reason in ('no eligible free GPU', 'no free GPU', 'no admissible placement')):
                            raise ManagerError(409, 'No safe placement is available for another chat; existing chats can continue') from exc
                        raise
                    service = (await self.manager.rig.status()).get('shared_kv_service') or {}
                    if service.get('instance_id') != instance:
                        raise ManagerError(409, 'Shared service changed during admission')
                    chats = {c['chat_id']: c for c in service.get('chats', [])}
                with self.db() as con:
                    con.execute('INSERT OR REPLACE INTO chat_bindings VALUES (?,?,?,?)', (owner, spec['model'], instance, chat))
            row = chats.get(chat)
            if not row:
                raise ManagerError(502, 'Shared context missing after admission')
            return {'profile': profile, 'state': 'ready', 'context_length': row.get('context_cap'),
                    'gpu_ids': [row['gpu']], 'route': spec['prefix'] + chat, 'chat_id': chat,
                    'instance_id': instance, 'used_tokens': row.get('used_tokens', 0),
                    'reservation': row.get('reservation')}

    async def compact(self, owner, expected_instance, expected_chat,
                      expected_used_tokens=None):
        """Clear one owner's KV context without releasing its GPU placement."""
        if (not isinstance(owner, str) or not owner or
                not isinstance(expected_instance, str) or not expected_instance or
                not isinstance(expected_chat, str) or not expected_chat or
                (expected_used_tokens is not None and
                 (type(expected_used_tokens) is not int or expected_used_tokens < 0))):
            raise ManagerError(400, 'Provide owner and exact shared context identity')
        async with self.lock:
            with self.db() as con:
                binding = con.execute(
                    'SELECT model, instance, chat FROM chat_bindings WHERE owner=?',
                    (owner,)).fetchone()
            if not binding or binding[1:] != (expected_instance, expected_chat):
                raise ManagerError(409, 'Chat placement changed; refresh before compaction')
            model = binding[0]
            service = (await self.manager.rig.status()).get('shared_kv_service') or {}
            if (service.get('status') != 'ready' or
                    service.get('instance_id') != expected_instance or
                    service.get('model') != model):
                raise ManagerError(409, 'Shared service changed before compaction')
            row = next((item for item in service.get('chats', [])
                        if item.get('chat_id') == expected_chat), None)
            if not row or type(row.get('used_tokens')) is not int:
                raise ManagerError(409, 'Shared chat context is missing')
            used = row['used_tokens']
            if expected_used_tokens is not None and used != expected_used_tokens:
                raise ManagerError(409, 'Shared chat changed before compaction')
            await self.manager._require_control()
            return await self.manager.rig.shared_kv(
                'reset', model=model, chat_id=expected_chat,
                expected_used_tokens=used)

    async def release(self, owner, expected_instance=None, expected_chat=None, park=False):
        if park and (not isinstance(expected_instance, str) or not expected_instance or
                     not isinstance(expected_chat, str) or not expected_chat):
            raise ManagerError(400, 'Parking requires the expected context and service identity')
        async with self.lock:
            with self.db() as con:
                row = con.execute('SELECT model, instance, chat FROM chat_bindings WHERE owner=?', (owner,)).fetchone()
            if not row:
                return {'state': 'cold'}
            model, instance, chat = row
            if expected_instance is not None and chat == 'main':
                raise ManagerError(409, 'The main chat is pinned; park an attached chat instead')
            if expected_instance is not None and (instance != expected_instance or chat != expected_chat):
                raise ManagerError(409, 'Chat placement changed; refresh before releasing')
            service = (await self.manager.rig.status()).get('shared_kv_service') or {}
            if chat != 'main' and service.get('instance_id') == instance and service.get('model') == model:
                await self.manager._require_control()
                await self.manager.rig.shared_kv('detach', model=model, chat_id=chat)
            with self.db() as con:
                con.execute('DELETE FROM chat_bindings WHERE owner=?', (owner,))
            return {'state': 'cold'}

    async def reset(self, target, expected, *, unload=False, expected_instance=None):
        async with self.lock:
            loaded, profiles, shared, service = await self.loaded()
            if not isinstance(expected, list) or sorted(expected) != sorted(loaded):
                raise ManagerError(409, 'Loaded models changed; refresh before changing the runtime')
            if unload:
                if target not in loaded:
                    raise ManagerError(409, 'Requested model is no longer loaded')
                if target == shared:
                    if not expected_instance or expected_instance != service.get('instance_id'):
                        raise ManagerError(409, 'Shared model instance changed; refresh before unloading')
                    await self.manager._require_control()
                    await self.manager.rig.shared_kv('stop', model=service['model'])
                else:
                    await self.manager.unload_profile(target)
                return {'unloaded': target, 'remaining': [p for p in loaded if p != target]}
            choices = self.offers()
            if target not in choices or any(p not in choices for p in loaded):
                raise ManagerError(409, 'Only listed chat models can be reset through chat controls')
            if target in loaded:
                return {'reset': [], 'target': target}
            if shared:
                await self.manager._require_control()
                await self.manager.rig.shared_kv('stop', model=service['model'])
            for pid in profiles:
                await self.manager.unload_profile(pid)
            return {'reset': loaded, 'target': target}

    async def switch(self, target, owner, reset_loaded=False):
        if target not in self.offers():
            raise ManagerError(404, 'Unknown local chat offer')
        loaded, _, _, _ = await self.loaded()
        others = [p for p in loaded if p != target]
        if others and not reset_loaded:
            raise ManagerError(409, 'Loaded models must be reset explicitly before switching')
        if others:
            await self.reset(target, loaded)
        await self.release(owner)
        return {'target': target}
