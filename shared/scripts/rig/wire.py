"""Versioned transport for supported rig CLI operations; never arbitrary host commands."""
import hashlib,json,os
from pathlib import Path
import sqlite3
import subprocess,sys
SCHEMA='rig-cli/1'
COMMANDS={'status','observe','start','stop','lease','claim','controller','plan','run','kill',
          'model-fit','schedule-preview','fit-observation','shared-kv','inspect','api'}
READ_ONLY={('status',),('observe',),('model-fit',),('schedule-preview',),('inspect',),
           ('lease','list'),('lease','preview'),('claim','list'),('controller','show'),
           ('plan','show'),('run','list'),('shared-kv','status')}


def execute(request, *, root, runner=None):
    response={'schema':SCHEMA,'ok':False}
    if not isinstance(request,dict) or request.get('schema')!=SCHEMA:
        return dict(response,kind='version_mismatch',message='unsupported rig CLI version')
    args=request.get('arguments');request_id=request.get('request_id')
    if not isinstance(request_id,str) or not request_id or len(request_id)>96:
        return dict(response,kind='invalid',message='request_id required')
    response['request_id']=request_id
    if not isinstance(args,list) or not args or not all(isinstance(v,str) and '\x00' not in v for v in args) or args[0] not in COMMANDS or args[:2]==['plan','run']:
        return dict(response,kind='unsupported_operation',message='unsupported rig command')
    response['arguments']=args
    read_only=any(tuple(args[:len(prefix)])==prefix for prefix in READ_ONLY)
    db=None
    if not read_only:
        root=Path(root);root.mkdir(parents=True,exist_ok=True,mode=0o700)
        db=sqlite3.connect(root/'wire.sqlite3',timeout=10)
        db.execute('CREATE TABLE IF NOT EXISTS requests (id TEXT PRIMARY KEY,digest TEXT,response TEXT)')
        digest=hashlib.sha256(json.dumps(args,separators=(',',':')).encode()).hexdigest()
        db.execute('BEGIN IMMEDIATE')
        old=db.execute('SELECT digest,response FROM requests WHERE id=?',(request_id,)).fetchone()
        if old:
            db.close()
            return json.loads(old[1]) if old[0]==digest else dict(response,kind='id_conflict',message='request ID reused')
        pending=dict(response,kind='uncertain',message='Dispatch may have happened; inspect owner state before a new request')
        db.execute('INSERT INTO requests VALUES (?,?,?)',(request_id,digest,json.dumps(pending)));db.commit()
    if runner is None:
        def runner(arguments):
            return subprocess.run([sys.executable,str(Path(__file__).with_name('rig.py')),*arguments],capture_output=True,text=True)
    try:
        result=runner(args)
        response.update(ok=result.returncode==0,exit_code=result.returncode,output=result.stdout,
                        diagnostics=result.stderr[-4000:])
        if db:
            db.execute('UPDATE requests SET response=? WHERE id=?',(json.dumps(response),request_id));db.commit()
        return response
    finally:
        if db:db.close()


def command(raw,root):
    try:request=json.loads(raw)
    except ValueError:request=None
    result=execute(request,root=root)
    print(json.dumps(result))
    return 0 if result['ok'] else result.get('exit_code',1)
