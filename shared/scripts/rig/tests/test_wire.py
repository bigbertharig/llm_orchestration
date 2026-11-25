from pathlib import Path
import sys
from types import SimpleNamespace
sys.path.insert(0,str(Path(__file__).parents[1]))
import wire


def test_wire_identity_idempotency_and_unknown_versions(tmp_path):
    calls=[]
    def runner(args):calls.append(args);return SimpleNamespace(returncode=0,stdout='{"lease":{}}',stderr='')
    packet={'schema':'rig-cli/1','request_id':'r1','arguments':['lease','release','lease-id','--owner','owner with spaces']}
    first=wire.execute(packet,root=tmp_path,runner=runner)
    assert first['ok'] and first['arguments']==packet['arguments']
    assert wire.execute(packet,root=tmp_path,runner=runner)==first
    assert len(calls)==1
    assert wire.execute(dict(packet,arguments=['stop']),root=tmp_path,runner=runner)['kind']=='id_conflict'
    assert wire.execute(dict(packet,schema='future'),root=tmp_path,runner=runner)['kind']=='version_mismatch'
    assert wire.execute(dict(packet,arguments=['bash','-c','anything']),root=tmp_path,runner=runner)['kind']=='unsupported_operation'


def test_read_only_wire_has_no_receipt_state_write(tmp_path):
    result=wire.execute({'schema':'rig-cli/1','request_id':'r','arguments':['status','--json']},root=tmp_path/'absent',runner=lambda args:SimpleNamespace(returncode=0,stdout='{}',stderr=''))
    assert result['ok']
    assert not (tmp_path/'absent').exists()
