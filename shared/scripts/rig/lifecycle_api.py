"""Versioned operator API. The rig owns admission, executable profiles and receipts."""
from __future__ import annotations
import argparse
from contextlib import redirect_stdout
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import re
import sqlite3

SCHEMA = 'rig-lifecycle/1'
OPERATIONS = ('capabilities', 'observe', 'profile.list', 'profile.register', 'controller.claim',
              'controller.heartbeat', 'controller.release', 'load', 'stop', 'lease.acquire',
              'lease.renew', 'lease.release', 'lease.preview', 'power.set', 'power.restore', 'claim.add', 'claim.renew', 'claim.remove')


def identifier(value):
    if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,95}', value):
        raise ValueError('invalid identifier')
    return value


class API:
    def __init__(self, rig, root=None):
        self.rig = rig
        self.root = Path(root or os.environ.get('RIG_LOCAL_STATE') or rig.HW_STATE_ROOT / 'local-api')

    def connect(self):
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        db = sqlite3.connect(self.root / 'operations.sqlite3', timeout=10)
        db.execute('CREATE TABLE IF NOT EXISTS requests (id TEXT PRIMARY KEY, digest TEXT, receipt TEXT)')
        db.execute('CREATE TABLE IF NOT EXISTS power_sessions (id TEXT PRIMARY KEY, controller TEXT, gpu INTEGER UNIQUE, previous TEXT)')
        db.execute('CREATE TABLE IF NOT EXISTS profiles (id TEXT PRIMARY KEY, spec TEXT)')
        return db

    def request(self, request):
        receipt = {'schema': SCHEMA, 'ok': False, 'observed_at': datetime.now(timezone.utc).isoformat()}
        try:
            if not isinstance(request, dict) or request.get('schema') != SCHEMA:
                raise ValueError('unsupported lifecycle schema')
            operation = request.get('operation')
            if operation not in OPERATIONS:
                raise ValueError('unsupported operation')
            receipt['operation'] = operation
            if operation == 'capabilities':
                return dict(receipt, ok=True, operations=list(OPERATIONS), profile_source='local_configuration',
                            controller_policy='explicit_claim', resume_policy='reconcile_uncertain_no_replay')
            if operation == 'observe':
                return dict(receipt, ok=True, observation=self.rig.status_payload())
            if operation == 'profile.list':
                with self.connect() as db:
                    return dict(receipt, ok=True, profiles=[json.loads(row[0]) for row in db.execute('SELECT spec FROM profiles ORDER BY id')])
            request_id = identifier(request.get('request_id'))
            receipt['request_id'] = request_id
            digest = hashlib.sha256(json.dumps(request, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
            with self.connect() as db:
                db.execute('BEGIN IMMEDIATE')
                prior = db.execute('SELECT digest,receipt FROM requests WHERE id=?', (request_id,)).fetchone()
                if prior:
                    if prior[0] != digest:
                        raise ValueError('request_id already used for different arguments')
                    return json.loads(prior[1])
                pending = dict(receipt, kind='uncertain', message='Operation started; observe owner state before retrying with a new ID')
                db.execute('INSERT INTO requests VALUES (?,?,?)', (request_id, digest, json.dumps(pending)))
            try:
                result = self.dispatch(operation, request)
                receipt.update(ok=True, **result)
            except Exception as exc:
                receipt.update(kind=getattr(exc, 'kind', 'refused'), message=str(exc))
            with self.connect() as db:
                db.execute('UPDATE requests SET receipt=? WHERE id=?', (json.dumps(receipt), request_id))
            return receipt
        except (ValueError, TypeError, KeyError) as exc:
            return dict(receipt, kind='invalid', message=str(exc))
        except (OSError, sqlite3.Error) as exc:
            return dict(receipt, kind='unavailable', message=type(exc).__name__)

    def dispatch(self, operation, request):
        rig = self.rig
        if operation == 'profile.register':
            pid = identifier(request.get('profile_id'))
            spec = request.get('spec')
            if not isinstance(spec, dict) or not isinstance(spec.get('model'), str) or not spec['model']:
                raise ValueError('spec requires a model identity')
            if type(spec.get('ctx_size')) is not int or spec['ctx_size'] <= 0:
                raise ValueError('spec requires positive ctx_size')
            model_path = Path(spec.get('gguf_path', ''))
            if not model_path.is_absolute() or not model_path.is_file():
                raise ValueError('gguf_path must be an existing absolute file on this host')
            resolved = rig.spec_profile(pid, spec)
            resolved['provenance'] = {'kind': 'local_configuration', 'evidence': None}
            with self.connect() as db:
                db.execute('BEGIN IMMEDIATE')
                # Registration is immutable; create a new profile ID for a changed configuration.
                prior = db.execute('SELECT spec FROM profiles WHERE id=?', (pid,)).fetchone()
                if prior and json.loads(prior[0]) != resolved:
                    raise ValueError('profile already registered with different settings')
                db.execute('INSERT OR IGNORE INTO profiles VALUES (?,?)', (pid, json.dumps(resolved)))
            return {'profile': resolved}
        controller = identifier(request.get('controller'))
        plan = rig.open_plan()
        if operation == 'controller.claim':
            return {'controller': plan.claim(controller, rig.caller(), False)}
        if operation == 'controller.heartbeat':
            return {'controller': plan.heartbeat(controller, rig.caller())}
        if operation == 'controller.release':
            plan.release(controller)
            return {'released': controller}
        active = plan.report().get('controller')
        if not active or active.get('id') != controller:
            raise rig.hw_plan.PlanError('not_controller', 'Claim controller ownership before mutation')
        plan.require_active(controller)
        if operation in ('power.set', 'power.restore', 'claim.add', 'claim.renew', 'claim.remove'):
            import power_session
            import hw_power
            with rig.hw_state.FileLock(str(rig.HW_STATE_ROOT / 'power-api.lock'), timeout=10):
                if operation == 'power.set':
                    gpu = request.get('gpu_id')
                    held = [row for row in rig.open_hw_state().leases() if gpu in row.get('gpu_ids', [])]
                    if not held or any(row.get('owner') != controller for row in held):
                        raise ValueError('power changes require a GPU lease owned by this controller')
                    path = hw_power.OVERRIDE_PATH
                    current = json.loads(path.read_text()) if path.is_file() else {}
                    previous = current.get('by_index', {}).get(str(gpu))
                    with self.connect() as db:
                        db.execute('INSERT INTO power_sessions VALUES (?,?,?,?)',
                                   (request['request_id'],controller,gpu,json.dumps(previous)))
                    # Keep the restoration record even when applying the cap fails halfway.
                    result = power_session.apply(gpu, request.get('watts'))
                    return dict(result, power_session_id=request['request_id'])
                session_id = identifier(request.get('power_session_id'))
                with self.connect() as db:
                    row = db.execute('SELECT controller,gpu,previous FROM power_sessions WHERE id=?', (session_id,)).fetchone()
                if not row or row[0] != controller:
                    raise ValueError('power session absent or owned by another controller')
                result = power_session.apply(row[1], json.loads(row[2]))
                with self.connect() as db:
                    db.execute('DELETE FROM power_sessions WHERE id=?', (session_id,))
                return dict(result, restored=session_id)
        if operation.startswith('claim.'):
            action = operation.split('.')[1]
            args = argparse.Namespace(action=action, name=request.get('name'),
                       gpus=','.join(map(str,request.get('gpu_ids',[]))),
                       note=request.get('note',''), by=controller, owner=request.get('owner',controller),
                       ttl=request.get('ttl_s',600), purpose=request.get('purpose','benchmark reservation'),
                       model=request.get('model',''), controller=controller)
            out = io.StringIO()
            with redirect_stdout(out): code = rig.cmd_claim(args)
            result = json.loads(out.getvalue())
            if code: raise rig.hw_state.StateError(result.get('kind','refused'),result.get('error','claim refused'))
            return result
        if operation.startswith('lease.'):
            action = operation.split('.')[1]
            args = argparse.Namespace(action=action, spec=json.dumps(request.get('spec', {})),
                                      controller=controller, lease_id=request.get('lease_id'),
                                      owner=request.get('owner', controller), ttl=request.get('ttl_s', 600), force=False)
            out = io.StringIO()
            with redirect_stdout(out):
                code = rig.cmd_lease(args)
            result = json.loads(out.getvalue())
            if code:
                raise rig.hw_state.StateError(result.get('kind', 'refused'), result.get('error', 'Lease refused'))
            return result
        pid = identifier(request.get('profile_id'))
        if operation == 'stop':
            loaded = rig.read_state().get('profiles', {}).get(pid)
            if not loaded:
                return {'stopped': pid, 'already_stopped': True}
            leases = rig.open_hw_state()
            lease = next((row for row in leases.leases() if row['lease_id'] == loaded.get('lease')), None)
            if lease and lease.get('owner') != controller:
                raise ValueError('profile lease belongs to another owner')
            with redirect_stdout(io.StringIO()):
                code = rig.cmd_stop(argparse.Namespace(profiles=[pid]))
            if code:
                raise ValueError('runtime stop failed; lease retained')
            if lease:
                leases.release(lease['lease_id'], controller)
            return {'stopped': pid}
        if operation != 'load':
            raise ValueError('unsupported operation')
        with self.connect() as db:
            row = db.execute('SELECT spec FROM profiles WHERE id=?', (pid,)).fetchone()
        if not row:
            raise ValueError('unknown local profile')
        spec = json.loads(row[0])
        gpu_ids = request.get('gpu_ids')
        if not isinstance(gpu_ids, list) or not gpu_ids or any(type(v) is not int or v < 0 for v in gpu_ids) or len(set(gpu_ids)) != len(gpu_ids):
            raise ValueError('gpu_ids must be distinct nonnegative integers')
        leases = rig.open_hw_state()
        lease_spec = {'controller': 'interactive', 'owner': controller, 'run_id': request['request_id'],
                      'gpu_ids': gpu_ids, 'ttl_s': 3600, 'metadata': {'profile_id': pid, 'controller_id': controller}}
        with rig.hw_state.FileLock(str(rig.HW_STATE_ROOT / 'admission.lock'), timeout=10):
            admitted = rig.admit_lease_spec(lease_spec, leases)
            lease = leases.acquire(admitted)
        try:
            args = argparse.Namespace(spec=json.dumps(spec), profiles=[pid], gpus=','.join(map(str,gpu_ids)),
                                      port=admitted['metadata']['port'], lease=lease['lease_id'],
                                      cache_ram=None, detach=False, queue=False, force=False)
            with redirect_stdout(io.StringIO()):
                code = rig.cmd_start_remote(args)
            if code:
                raise ValueError('runtime load failed')
            return {'profile_id': pid, 'lease': lease, 'observation': rig.status_payload()}
        except BaseException:
            # Never release admission while a partially launched runtime may still own the cards.
            if pid not in rig.read_state().get('profiles', {}):
                leases.release(lease['lease_id'], controller)
            raise


def command(rig, raw):
    try:
        value = json.loads(raw)
    except ValueError as exc:
        value = None
    receipt = API(rig).request(value)
    print(json.dumps(receipt))
    return 0 if receipt['ok'] else 1
