from pathlib import Path
import sys
from types import SimpleNamespace
import pytest
sys.path.insert(0,str(Path(__file__).parents[1]))
import lifecycle_api
import hw_plan
import hw_state


class FakeRig:
    hw_plan=hw_plan
    hw_state=hw_state
    def __init__(self,root):
        self.HW_STATE_ROOT=root; self.loaded={}; self.fail=False
        self.hardware=hw_state.HardwareState(root)
    def open_plan(self): return hw_plan.Plan(self.HW_STATE_ROOT)
    def caller(self): return 'fixture'
    def open_hw_state(self): return self.hardware
    def status_payload(self): return {'profiles':list(self.loaded.values())}
    def read_state(self): return {'profiles':self.loaded}
    def spec_profile(self,pid,spec): return dict(spec,id=pid)
    def admit_lease_spec(self,spec,hs): return dict(spec,ports=[11434],metadata={'port':11434})
    def cmd_start_remote(self,args):
        if self.fail: return 1
        self.loaded[args.profiles[0]]={'id':args.profiles[0],'lease':args.lease,'state':'ready'}
        return 0
    def cmd_stop(self,args):
        for pid in args.profiles: self.loaded.pop(pid,None)
        return 0


def test_local_model_complete_lifecycle_without_registry(tmp_path):
    rig=FakeRig(tmp_path/'hw');api=lifecycle_api.API(rig,tmp_path/'api')
    model=tmp_path/'outside.gguf';model.write_bytes(b'fixture')
    count=0
    def request(operation,**kw):
        nonlocal count;count+=1
        return api.request(dict({'schema':lifecycle_api.SCHEMA,'request_id':str(count),'controller':'local','operation':operation},**kw))
    profile={'model':'user/model','gguf_path':str(model),'ctx_size':4096}
    assert request('profile.register',profile_id='user',spec=profile)['ok']
    assert request('profile.list')['profiles'][0]['provenance']['evidence'] is None
    assert request('load',profile_id='user',gpu_ids=[0])['kind']=='not_controller'
    assert request('controller.claim')['ok']
    other=request('controller.claim',controller='other')
    assert other['kind']=='not_controller'
    loaded=request('load',profile_id='user',gpu_ids=[0]); assert loaded['ok'],loaded
    assert rig.hardware.leases()
    duplicate=api.request(dict(schema=lifecycle_api.SCHEMA,request_id=str(count),controller='local',operation='load',profile_id='user',gpu_ids=[0]))
    assert duplicate==loaded
    assert request('stop',profile_id='user')['ok']
    assert not rig.hardware.leases() and not rig.loaded
    rig.fail=True
    assert not request('load',profile_id='user',gpu_ids=[0])['ok']
    assert not rig.hardware.leases()
    assert not request('profile.register',profile_id='user',spec=dict(profile,ctx_size=8192))['ok']


def test_unknown_version_and_uncertain_id_never_dispatch(tmp_path):
    rig=FakeRig(tmp_path/'hw');api=lifecycle_api.API(rig,tmp_path/'api')
    assert api.request({'schema':'future'})['kind']=='invalid'
    import hashlib,json
    packet={'schema':lifecycle_api.SCHEMA,'request_id':'pending','operation':'controller.claim','controller':'local'}
    digest=hashlib.sha256(json.dumps(packet,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    with api.connect() as db:
        db.execute('INSERT INTO requests VALUES (?,?,?)',('pending',digest,json.dumps({'schema':lifecycle_api.SCHEMA,'ok':False,'kind':'uncertain'})))
    assert api.request(packet)['kind']=='uncertain'
    assert rig.open_plan().report()['controller'] is None


def test_power_bounds_and_restore_preserve_other_cards(tmp_path,monkeypatch):
    import power_session
    monkeypatch.setattr(power_session.hw_power,'cards',lambda:[{'index':0,'min_w':50,'max_w':150}])
    path=tmp_path/'power.json';path.write_text('{"by_index":{"1":80}}')
    with pytest.raises(ValueError): power_session.apply(0,999,override=path,restart=lambda:None)
    power_session.apply(0,100,override=path,restart=lambda:None)
    power_session.apply(0,None,override=path,restart=lambda:None)
    assert __import__('json').loads(path.read_text())=={'by_index':{'1':80}}


def test_live_power_session_uses_installed_helper_and_restores_base_plan(monkeypatch):
    import power_session
    from types import SimpleNamespace
    monkeypatch.setattr(power_session.hw_power, 'cards', lambda: [
        {'index': 0, 'min_w': 100, 'max_w': 480, 'default_w': 450, 'name': 'NVIDIA GeForce RTX 3090 Ti'}])
    monkeypatch.setattr(power_session.hw_power, 'load_rules', lambda **kw: {'by_index': {}, 'by_model': {'RTX 3090 Ti':350}})
    calls=[]
    def run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=0, stdout='{"gpu_id":0,"previous_override_w":300,"override_w":null,"applied_w":350}', stderr='')
    monkeypatch.setattr(power_session.subprocess, 'run', run)
    assert power_session.apply(0,None)['applied_w']==350
    assert calls[0]==['sudo','-n','/usr/local/libexec/rig-power-override','0','restore','350.0']
