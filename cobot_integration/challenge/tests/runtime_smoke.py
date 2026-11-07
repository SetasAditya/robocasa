"""Bounded development-only checks; never counts as a successful policy campaign.

python -m cobot_integration.challenge.tests.runtime_smoke --output NEWDIR --jobs 3
"""
import argparse
import concurrent.futures
import json
from pathlib import Path
import subprocess
import sys


def worker(name, folder):
    import numpy as np
    from PIL import Image
    from ..catalog import recipe
    from ..runner import prepare, qualify, load, replay, hold, SensorInterface, render_camera
    prepare(folder, recipe(name))
    evidence = qualify(folder)
    env, _ = load(folder)
    try:
        Image.fromarray(render_camera(env, 'mobilebase0_kinect')).save(folder/'kinect.jpg', quality=78)
        if name == 'ShallowDrawerRetrieval':
            obs = SensorInterface(env).observe()
            assert obs['kinect_rgb'].shape == (360,640,3)
            assert obs['kinect_depth_m'].shape == (360,640)
            assert not ({'objects','slot','success','fixture_poses'} & set(obs))
            actions = np.array([hold(env) for _ in range(10)])
    finally:
        env.close()
    if name == 'ShallowDrawerRetrieval':
        a, x, s = replay(folder, actions)
        b, y, t = replay(folder, actions)
        assert np.array_equal(actions, np.zeros_like(actions))
        assert a == b and s == t and np.allclose(x, y, atol=1e-9, rtol=0)
        evidence.update(sensor_contract_passed=True, independent_action_replay_passed=True)
    print('RESULT='+json.dumps(evidence), flush=True)


def main():
    from ..catalog import PROFILES
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--jobs',type=int,default=3)
    p.add_argument('--worker',choices=PROFILES)
    a=p.parse_args()
    if a.worker:
        worker(a.worker,a.output);return
    if a.output.exists():raise FileExistsError(a.output)
    if a.jobs<1:raise ValueError('jobs must be positive')
    a.output.mkdir(parents=True)
    def run(name):
        if name=='ImperfectGraspRecovery':
            return dict(task=name,status='requires_physical_grasp_initialization',qualified=False)
        try:
            result=subprocess.run([sys.executable,'-m',__spec__.name,'--worker',name,'--output',str(a.output/name)],
                                  capture_output=True,text=True,timeout=240)
            (a.output/(name+'.log')).write_text(result.stdout+result.stderr)
            lines=[s[7:] for s in result.stdout.splitlines() if s.startswith('RESULT=')]
            data=dict(task=name,status='validated_reset' if result.returncode==0 else 'failed',
                      result=json.loads(lines[-1]) if lines else {},tail=result.stderr[-1600:] if result.returncode else '')
        except subprocess.TimeoutExpired as exc:
            (a.output/(name+'.log')).write_text(str(exc.stdout)+str(exc.stderr))
            data=dict(task=name,status='timeout')
        print(json.dumps(data),flush=True)
        return data
    with concurrent.futures.ThreadPoolExecutor(max_workers=a.jobs) as pool:
        results=list(pool.map(run,PROFILES))
    (a.output/'summary.json').write_text(json.dumps(results,indent=2)+'\n')
    if any(r['status'] in ('failed','timeout') for r in results):raise SystemExit(1)


if __name__=='__main__':main()
