import asyncio,json,sys
from pathlib import Path
import pytest
from rig_control.rig import Rig,RigUnavailable


def test_wire_roundtrip_and_unknown_receipt(tmp_path):
    script=tmp_path/'runtime.py'
    script.write_text("import json,sys\nr=json.loads(sys.argv[3])\nprint(json.dumps(dict(r,ok=True,exit_code=0,output='{\"profiles\":[]}',diagnostics='')))\n")
    rig=Rig((sys.executable,str(script)),30,wire_version='rig-cli/1')
    assert asyncio.run(rig.status())=={'profiles':[]}
    script.write_text("print('{\"schema\":\"future\"}')\n")
    with pytest.raises(RigUnavailable,match='mismatch'):asyncio.run(rig.status())
