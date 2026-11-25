"""Authenticated local-runtime authority. Run independently with rig-control serve."""
from __future__ import annotations
import argparse
import asyncio
import fcntl
import hmac
import json
import os
import re
import shlex
import sqlite3
from contextlib import asynccontextmanager
from pathlib import Path
from uuid import uuid4
import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from .manager import Manager, ManagerError
from .manager import models
from .manager.routes import router
from .profiles import Profiles
from .rig import Rig, RigUnavailable
from .chat_runtime import ChatRuntime, SHARED_PROFILES
from .shared_chat_contract import ROUTE_PREFIX_BY_MODEL

SCHEMA = 'orchestration-control/1'
MANAGER_READS = {'status', 'chat_availability', 'leases', 'claims', 'plan', 'runs', 'observe',
                 'preview_lease', 'fit_model', 'schedule_preview'}
MANAGER_WRITES = {'unload_profile', 'load_model', 'kill', 'remove_claim', 'recover_run',
                  'dismiss_run', 'stop_run', 'cancel_item'}


def create_app(*, token, state_root, rig=None, profiles_path=None, controller_id='orchestration',
               start_timeout=900, reconcile=True, model_prefix='gpu-rig'):
    if not token:
        raise ValueError('ORCHESTRATION_CONTROL_TOKEN is required')
    root = Path(state_root)
    root.mkdir(parents=True, exist_ok=True)
    def db():
        con = sqlite3.connect(root / 'requests.sqlite3', timeout=30)
        con.execute('CREATE TABLE IF NOT EXISTS requests (id TEXT PRIMARY KEY, payload TEXT, state TEXT, response TEXT)')
        return con
    with db():
        pass
    rig = rig or Rig(tuple(shlex.split(os.getenv('ORCHESTRATION_RIG_COMMAND', 'rig'))), start_timeout, wire_version='rig-cli/1')
    manager = Manager(rig, start_timeout, controller_id, mirror_path=root / 'rig_report.json')
    source = Profiles(Path(profiles_path or Path.home() / '.config/llm-orchestration/remote_profiles.json'))
    def profiles():
        if not source.path.exists():
            return {}
        return source.all()
    chat = ChatRuntime(manager, profiles, db)
    tasks = set()
    @asynccontextmanager
    async def lifespan(app):
        with (root / 'service.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            worker = asyncio.create_task(manager.run_reconciler()) if reconcile else None
            try:
                yield
            finally:
                if worker:
                    worker.cancel()
                    await asyncio.gather(worker, return_exceptions=True)
                if tasks:
                    await asyncio.gather(*list(tasks), return_exceptions=True)
    app = FastAPI(lifespan=lifespan)
    app.state.manager = manager
    app.state.chat = chat
    internal = FastAPI()
    internal.include_router(router(manager, profiles, model_prefix))
    @internal.exception_handler(ManagerError)
    async def internal_error(request, exc):
        return JSONResponse({'error': {'message': str(exc), **exc.extra}}, status_code=exc.status)
    @internal.exception_handler(RigUnavailable)
    async def internal_rig_error(request, exc):
        return JSONResponse({'error': {'message': str(exc)}}, status_code=exc.status)

    async def dispatch(op, args):
        if op == 'capabilities':
            return {'schema': SCHEMA, 'local_runtime_authority': True, 'operations': sorted(MANAGER_READS | MANAGER_WRITES)}
        if op == 'catalog.role':
            from . import role_model
            return role_model.resolve_model(args['model']) if 'model' in args else role_model.resolve(args['role'])
        if op == 'snapshot':
            try:
                summary = models.catalog().get('summary') or {}
            except (OSError, ValueError):
                summary = {'models': [], 'roles': {}, 'key_tests': [], 'error': 'No optional model evidence imported'}
            return {'model_prefix': model_prefix, 'profiles': profiles(), 'chat_offers': chat.offers(), 'summary': summary,
                    'shared_profiles': SHARED_PROFILES, 'route_prefixes': ROUTE_PREFIX_BY_MODEL}
        if op == 'rig.status':
            return await rig.status()
        if op == 'route.prepare':
            pid = args.get('profile')
            if pid not in profiles():
                raise ManagerError(404, 'Unknown local runtime profile')
            if args.get('ready_only'):
                rows = (await rig.status()).get('profiles', [])
                if isinstance(rows, dict):
                    rows = [{'id': key, **value} for key, value in rows.items()]
                row = next((r for r in rows if r.get('id') == pid and r.get('state') == 'ready'), None)
                if (not row or row.get('model') != profiles()[pid].get('model') or
                        type(row.get('ctx_size')) is not int or row['ctx_size'] <= 0 or
                        type(row.get('port')) is not int or not 1 <= row['port'] <= 65535):
                    raise ManagerError(409, 'Requested local profile is not ready', kind='model_not_ready')
                return {'port': row['port']}
            port = await rig.ensure_ready(pid, start=lambda: manager.load_profile(pid, profiles()[pid]))
            return {'port': port}
        if op == 'shared.turn':
            # Local inference is the only raw shared-runtime operation exposed to clients.
            return await rig.shared_kv('turn', **args)
        if op == 'chat.prepare':
            return await chat.prepare(**args)
        if op == 'chat.release':
            return await chat.release(**args)
        if op == 'chat.switch':
            return await chat.switch(**args)
        if op == 'chat.reset':
            return await chat.reset(**args)
        if op.startswith('manager.') and op[8:] in MANAGER_READS | MANAGER_WRITES:
            return await getattr(manager, op[8:])(*args.get('args', []), **args.get('kwargs', {}))
        if op == 'manager.http':
            method, path = args.get('method'), args.get('path', '')
            if method not in ('GET', 'POST') or not re.fullmatch(r'/v1/manager/[A-Za-z0-9_./:-]+', path) or '..' in path:
                raise ManagerError(400, 'Invalid manager request')
            if method == 'POST' and not path.startswith('/v1/manager/controller/'):
                await manager._require_control()
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=internal), base_url='http://owner') as client:
                response = await client.request(method, path, params=args.get('query'), json=args.get('body'))
            if response.status_code >= 400:
                error = response.json().get('error') or {}
                raise ManagerError(response.status_code, error.pop('message', 'Runtime request refused'), **error)
            return {"status": response.status_code, "body": response.json()}
        raise ManagerError(400, 'Unknown Orchestration operation')

    def read_only(op, args):
        return (op in ('capabilities', 'snapshot', 'rig.status', 'catalog.role') or
                op.startswith('manager.') and op[8:] in MANAGER_READS or
                op == 'manager.http' and args.get('method') == 'GET' or
                op == 'route.prepare' and args.get('ready_only') is True)

    async def execute(envelope, payload, mutation):
        rid = envelope['request_id']
        response = {'schema': SCHEMA, 'request_id': rid}
        try:
            response.update(ok=True, result=await dispatch(envelope['operation'], envelope['arguments']))
        except (ManagerError, RigUnavailable) as exc:
            response.update(ok=False, error={'status': exc.status, 'message': str(exc), **getattr(exc, 'extra', {})})
        except (TypeError, ValueError, KeyError) as exc:
            response.update(ok=False, error={'status': 400, 'message': str(exc)})
        except Exception:
            # Leave the durable dispatch marker uncertain; callers reconcile this exact request.
            response.update(ok=False, error={'status': 502, 'kind': 'reconciliation_required',
                                            'message': 'Runtime operation interrupted; reconcile the original request ID'})
            return response
        if mutation:
            with db() as con:
                con.execute('UPDATE requests SET state=?,response=? WHERE id=?', ('complete', json.dumps(response), rid))
        return response

    @app.get('/health')
    async def health():
        return {'status': 'ok', 'schema': SCHEMA}

    @app.post('/v1/control')
    async def control(request: Request):
        supplied = request.headers.get('authorization', '').removeprefix('Bearer ')
        if not hmac.compare_digest(supplied.encode(), token.encode()):
            return JSONResponse({'error': 'Unauthorized'}, status_code=401)
        raw = await request.body()
        if len(raw) > 2_000_000:
            return JSONResponse({'error': 'Request too large'}, status_code=413)
        try:
            envelope = json.loads(raw)
            if not isinstance(envelope, dict) or envelope.get('schema') != SCHEMA:
                raise ValueError('Unsupported control schema')
            rid = envelope.get('request_id')
            if not isinstance(rid, str) or not re.fullmatch(r'[A-Za-z0-9_.:-]{1,128}', rid):
                raise ValueError('Bounded request_id is required')
            if not isinstance(envelope.get('operation'), str) or not isinstance(envelope.get('arguments'), dict):
                raise ValueError('operation and arguments are required')
        except (ValueError, TypeError) as exc:
            return JSONResponse({'error': str(exc)}, status_code=400)
        payload = json.dumps(envelope, sort_keys=True, separators=(',', ':'))
        mutation = not read_only(envelope['operation'], envelope['arguments'])
        if mutation:
            with db() as con:
                previous = con.execute('SELECT payload,state,response FROM requests WHERE id=?', (rid,)).fetchone()
                if previous:
                    if previous[0] != payload:
                        return {'schema': SCHEMA, 'request_id': rid, 'ok': False,
                                'error': {'status': 409, 'kind': 'id_collision', 'message': 'request_id was already used with different arguments'}}
                    if previous[1] == 'complete':
                        return json.loads(previous[2])
                    return {'schema': SCHEMA, 'request_id': rid, 'ok': False,
                            'error': {'status': 409, 'kind': 'reconciliation_required', 'message': 'Operation is running or interrupted; reconcile this request before retrying'}}
                con.execute('INSERT INTO requests VALUES (?,?,?,NULL)', (rid, payload, 'dispatching'))
        task = asyncio.create_task(execute(envelope, payload, mutation))
        tasks.add(task)
        task.add_done_callback(tasks.discard)
        return await asyncio.shield(task)
    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    serve = sub.add_parser('serve')
    serve.add_argument('--bind', default='127.0.0.1')
    serve.add_argument('--port', type=int, default=8092)
    serve.add_argument('--state-root', type=Path, default=Path.home() / '.local/state/llm-orchestration/control')
    serve.add_argument('--profiles', type=Path, default=Path.home() / '.config/llm-orchestration/remote_profiles.json')
    request = sub.add_parser('request')
    request.add_argument('operation')
    request.add_argument('--arguments', default='{}')
    request.add_argument('--request-id', default=None)
    args = parser.parse_args()
    token = os.environ.get('ORCHESTRATION_CONTROL_TOKEN', '')
    if not token:
        parser.error('ORCHESTRATION_CONTROL_TOKEN is required')
    if args.command == 'serve':
        import uvicorn
        uvicorn.run(create_app(token=token, state_root=args.state_root, profiles_path=args.profiles, model_prefix=os.getenv('ORCHESTRATION_MODEL_PREFIX', 'gpu-rig')), host=args.bind, port=args.port)
    else:
        rid = args.request_id or uuid4().hex
        response = httpx.post(os.environ.get('ORCHESTRATION_URL', 'http://127.0.0.1:8092') + '/v1/control',
            headers={'Authorization': 'Bearer ' + token}, timeout=900,
            json={'schema': SCHEMA, 'request_id': rid, 'operation': args.operation, 'arguments': json.loads(args.arguments)})
        response.raise_for_status()
        value = response.json()
        if value.get('schema') != SCHEMA or value.get('request_id') != rid:
            raise SystemExit('Control response identity/version mismatch')
        print(json.dumps(value, indent=2))
        raise SystemExit(0 if value.get('ok') else 1)

if __name__ == '__main__':
    main()
