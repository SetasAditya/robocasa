"""Closed-loop checkpoint evaluation via a private local socket; no recorded actions substituted."""
import os
from pathlib import Path
import socket
import subprocess
import tempfile
import time
import numpy as np
from .manifest import read,write,file_sha
from .recording import Recorder
from .runtime import restore
from .protocol import receive,send


def evaluate(scene,checkpoint,output,python,max_steps=1800,witness=None):
    checkpoint=Path(checkpoint).resolve();identity=read(checkpoint/'cobot_data.json')
    env,r=restore(scene);rec=None;process=None
    output=Path(output);clipped=0
    try:
        if identity['contract']!=r['contract']:raise ValueError('Checkpoint controller/camera contract mismatch')
        if identity['track']!=r['request']['track']:raise ValueError('Cannot mix experiment tracks')
        if r['request']['split']!='test':raise ValueError('Policy evaluation requires test scenes')
        if r['request']['track']=='challenge':
            from .validation import certified
            if witness is None:raise ValueError('Challenge evaluation requires a successful reviewed witness episode')
            wm,wv=certified(witness)
            if wm['scene_id']!=r['id'] or not wv['checks']['geometry_review']:
                raise ValueError('Witness does not qualify this exact challenge scene')
        rec=Recorder(env,scene,output,dict(kind='learned_policy',checkpoint=str(checkpoint),
                         checkpoint_sha256=file_sha(checkpoint/'model.safetensors')))
        with tempfile.TemporaryDirectory(prefix='cobot-policy-') as temp:
            address=Path(temp)/'policy.sock'
            environment=dict(os.environ);repo=Path(__file__).resolve().parents[2]
            environment['PYTHONPATH']=str(repo)+os.pathsep+environment.get('PYTHONPATH','')
            with (output/'policy.log').open('w') as log:
                process=subprocess.Popen([str(python),'-m','cobot_integration.data.policy_worker','--socket',str(address),'--checkpoint',str(checkpoint)],env=environment,stdout=log,stderr=subprocess.STDOUT)
                deadline=time.monotonic()+300
                while not address.exists():
                    if process.poll() is not None or time.monotonic()>deadline:raise RuntimeError('Policy worker unavailable; see policy.log')
                    time.sleep(.1)
                with socket.socket(socket.AF_UNIX,socket.SOCK_STREAM) as conn:
                    conn.settimeout(300);conn.connect(str(address))
                    hello,_=receive(conn)
                    if hello['contract']!=r['contract']:raise ValueError('Worker contract mismatch')
                    while rec.count<max_steps and not rec.last_success:
                        obs=rec.observe();arrays={k:v for k,v in obs.items() if k!='task'}
                        send(conn,dict(command='infer',instruction=r['instruction']),arrays)
                        response,packet=receive(conn)
                        if not response.get('ok'):raise ValueError('Policy inference failed')
                        for a in packet['action'][:min(5,max_steps-rec.count)]:
                            bounded=np.clip(a,np.asarray(r['contract']['low']),np.asarray(r['contract']['high']))
                            clipped+=int(np.count_nonzero(bounded!=a))
                            rec.step(bounded,'policy')
                            if rec.last_success:break
                    send(conn,dict(command='close'))
                process.wait(timeout=30)
        result=rec.close();write(output/'evaluation.json',dict(success=result['success'],steps=rec.count,controller='pi05',test_only=True,clipped_components=clipped))
        return result
    except BaseException:
        if rec is not None:rec.close('interrupted_or_failed')
        raise
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            try:process.wait(timeout=10)
            except subprocess.TimeoutExpired:process.kill();process.wait()
        env.close()
