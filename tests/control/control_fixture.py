"""HTTP test adapter for owner policy; no Gateway product imports."""
from types import SimpleNamespace
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from rig_control.manager import Manager, ManagerError
from rig_control.manager.routes import router
from rig_control.profiles import Profiles
from rig_control.rig import Rig, RigUnavailable
from fixture_settings import Settings

def create_app(settings, **kwargs):
    rig = Rig(settings.rig_command, settings.start_timeout_seconds)
    manager = Manager(rig, settings.start_timeout_seconds, getattr(settings, 'controller_id', 'controller'), getattr(settings, "mirror_path", None))
    profiles = Profiles(settings.profiles_path)
    app = FastAPI()
    app.state.manager = manager
    app.include_router(router(manager, profiles.all, settings.model_prefix))
    @app.exception_handler(ManagerError)
    @app.exception_handler(RigUnavailable)
    async def error(request, exc):
        return JSONResponse({'error': {'message': str(exc), **getattr(exc, 'extra', {})}}, status_code=exc.status)
    @app.post('/v1/chat/completions')
    async def load(request: Request):
        body = await request.json()
        pid = body['model'].split('/', 1)[1]
        port = await rig.ensure_ready(pid, start=lambda: manager.load_profile(pid, profiles.all()[pid]))
        return {'port': port}
    @app.post('/test/prepare/{profile}')
    async def prepare(profile: str):
        return await manager.prepare_chat_profile(profile, profiles.all()[profile])
    return app
