"""Scoped temporary power override owned and restored by Orchestration."""
import json
import math
import os
import subprocess
from pathlib import Path
import hw_power


def apply(gpu, watts, *, override=None, restart=None):
    if type(gpu) is not int or gpu < 0:
        raise ValueError('invalid GPU')
    cards = hw_power.cards()
    card = next((c for c in cards if c['index'] == gpu), None)
    if card is None:
        raise ValueError('GPU is absent')
    if watts is not None and (type(watts) not in (int,float) or not math.isfinite(watts) or not card['min_w'] <= watts <= card['max_w']):
        raise ValueError('watts outside hardware bounds')
    path = Path(override or hw_power.OVERRIDE_PATH)
    doc = json.loads(path.read_text()) if path.is_file() else {}
    values = doc.setdefault('by_index', {})
    previous = values.get(str(gpu))
    if watts is None:
        values.pop(str(gpu), None)
    else:
        values[str(gpu)] = watts
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.api.tmp')
    with temp.open('w') as out:
        json.dump(doc, out); out.flush(); os.fsync(out.fileno())
    os.replace(temp, path)
    if restart is None:
        restart = lambda: subprocess.run(['sudo','-n','systemctl','restart','rig-power.service'], check=True, capture_output=True, timeout=30)
    restart()
    return {'gpu_id': gpu, 'previous_override_w': previous, 'override_w': watts}
