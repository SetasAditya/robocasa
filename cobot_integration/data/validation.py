"""Raw row audits and independent action-only replay certificates."""
import io
import math
from pathlib import Path
import h5py
import numpy as np
from PIL import Image
from .manifest import VERSION, read, write, file_sha, check_request, identity
from .contract import action


def inspect_raw(directory):
    directory=Path(directory);m=read(directory/'episode.json');check_request(m['request'])
    if m['version']!=VERSION:raise ValueError('Dataset contract version mismatch')
    if file_sha(directory/'capture.h5')!=m.get('capture_sha256'):raise ValueError('Capture hash changed')
    if m.get('producer',{}).get('kind')=='expert_plugin':
        if file_sha(directory/'expert_source.py')!=m['producer']['sha256']:raise ValueError('Expert source snapshot changed')
    spec=m['contract']
    if identity({k:v for k,v in spec.items() if k!='hash'})!=spec['hash']:raise ValueError('Contract hash changed')
    n=m['frames']
    if n<1:raise ValueError('Empty episode')
    with h5py.File(directory/'capture.h5') as f:
        columns=('sim_state','state','action','timestamp','success','contact_violation','task_state','phase','kinect','wrist')
        if any(len(f[key])!=n for key in columns):raise ValueError('Sensor/action row count mismatch')
        if f['state'].shape!=(n,18) or f['action'].shape!=(n,spec['action_dim']):raise ValueError('Feature dimensions changed')
        for key in ('sim_state','state','action','timestamp','final_state'):
            if not np.isfinite(f[key][:]).all():raise ValueError('Nonfinite '+key)
        if not np.allclose(f['timestamp'][:],np.arange(n)/20,atol=1e-6,rtol=0):raise ValueError('Noncausal or wrong-rate timestamps')
        for a in f['action']:action(a,spec)
        for i in sorted({0,n//2,n-1}):
            for camera in ('kinect','wrist'):
                image=decode(f[camera][i])
                if image.shape!=(360,640,3):raise ValueError('Invalid decoded camera shape')
    return m


def decode(buffer):
    with Image.open(io.BytesIO(bytes(buffer))) as image:return np.array(image.convert('RGB'))


def same(a,b):
    if isinstance(a,dict):return isinstance(b,dict) and a.keys()==b.keys() and all(same(a[k],b[k]) for k in a)
    if isinstance(a,list):return isinstance(b,list) and len(a)==len(b) and all(same(x,y) for x,y in zip(a,b))
    if isinstance(a,bool) or a is None:return a==b
    if isinstance(a,(int,float)):return isinstance(b,(int,float)) and math.isclose(a,b,abs_tol=1e-5,rel_tol=0)
    return a==b


def validate(directory,review=None):
    import json
    from .runtime import restore,observation,task_state,contacts,begin_step
    directory=Path(directory);m=inspect_raw(directory);env,scene=restore(m['scene'])
    checks=dict(recording_completed=m['status']=='recorded',recorded_success=bool(m['success']),
                initial_scene=not scene['initial_contacts'] and not scene['initial_success'],
                scene_binding=m['scene_id']==scene['id'] and m['source_sha256']==scene['source_sha256'] and m['contract']==scene['contract'],
                timing=True,action_replay=True,state18_replay=True,task_history_replay=True,
                cameras=True,contacts=True,terminal_success=False,geometry_review=m['request']['track']=='transfer')
    state_error=0.;sensor_error=0.;rgb_error=0.
    try:
        with h5py.File(directory/'capture.h5') as f:
            n=m['frames'];sample={0,n//2,n-1}
            for i in range(n):
                obs=observation(env)
                begin_step(env)
                state_error=max(state_error,float(np.max(np.abs(env.sim.get_state().flatten()-f['sim_state'][i]))))
                sensor_error=max(sensor_error,float(np.max(np.abs(obs['observation.state']-f['state'][i]))))
                checks['task_history_replay'] &= same(json.loads(f['task_state'].asstr()[i]),task_state(env))
                if i in sample:
                    for camera in ('kinect','wrist'):
                        error=np.abs(obs['observation.images.'+camera].astype(float)-decode(f[camera][i]).astype(float)).mean()
                        rgb_error=max(rgb_error,float(error))
                env.step(action(f['action'][i],m['contract']))
                checks['contacts'] &= not bool(contacts(env)) and not bool(f['contact_violation'][i])
                last_success=bool(env._check_success())
                checks['action_replay'] &= last_success==bool(f['success'][i])
            state_error=max(state_error,float(np.max(np.abs(env.sim.get_state().flatten()-f['final_state'][:]))))
            checks['terminal_success']=last_success
        checks['action_replay'] &= state_error<=1e-5
        checks['state18_replay']=sensor_error<=1e-5
        checks['cameras']=rgb_error<=8.
        if review:
            r=read(review)
            checks['geometry_review']=bool(r.get('scene_id')==scene['id'] and r.get('source_sha256')==scene['source_sha256']
                and r.get('reviewer') and all(r.get(k) is True for k in ('collision_geometry_reviewed','primary_axis_verified','initialization_verified')))
        else:r=None
    finally:env.close()
    passed=all(checks.values())
    certificate=dict(version=VERSION,passed=passed,training_eligible=passed and m['request']['split']!='test',
        evaluation_eligible=passed,checks=checks,scene_id=m['scene_id'],source_sha256=m['source_sha256'],
        episode_sha256=file_sha(directory/'episode.json'),capture_sha256=m['capture_sha256'],
        max_state_error=state_error,max_state18_error=sensor_error,sampled_rgb_mae=rgb_error,review=r)
    # Preserve each validation attempt. Latest pointer is convenience, never the only evidence.
    import time
    write(directory/f'validation-{time.time_ns()}.json',certificate)
    write(directory/'validation.json',certificate,overwrite=True)
    return certificate


def certified(directory):
    directory=Path(directory);m=inspect_raw(directory);v=read(directory/'validation.json')
    required={'recording_completed','recorded_success','initial_scene','scene_binding','timing','action_replay','state18_replay','task_history_replay','cameras','contacts','terminal_success','geometry_review'}
    if (v.get('passed') is not True or set(v.get('checks',{}))!=required
            or not all(v['checks'].values()) or m['status']!='recorded' or not m['success'] or m.get('contact_violation_steps',0)):
        raise ValueError('Episode not replay-certified')
    if v['episode_sha256']!=file_sha(directory/'episode.json') or v['capture_sha256']!=m['capture_sha256'] or v['scene_id']!=m['scene_id'] or v['source_sha256']!=m['source_sha256']:
        raise ValueError('Stale replay certificate')
    return m,v
