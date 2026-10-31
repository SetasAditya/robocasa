import os
os.environ.setdefault('MUJOCO_GL', 'egl')
import argparse, json, signal, traceback
from pathlib import Path
from kitchen_tasks import environment_config
from .experts import LabAptExpert
from moma_validation import EpisodeRecorder, validate_replay
from . import object_snapshot as snapshot
import robosuite as suite
p = argparse.ArgumentParser()
p.add_argument('--seed', type=int, default=0)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=False)
from . import ROOT, snapshot_sources
snapshot_sources(a.output)
config = environment_config('fetch_surface', a.seed, 100, 1, render=False, images=False, obj_groups=str(ROOT / 'robocasa/models/assets/objects/objaverse/can/can_0/model.xml'))
r = dict(task='fetch_surface', seed=a.seed, success=False, accepted=False, training_eligible=False)
env = None
rec = None
x = None
try:

    def timeout(*_):
        raise TimeoutError('300 second probe limit')
    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(300)
    env = suite.make(**config)
    env.reset()
    rec = EpisodeRecorder(env, snapshot, lambda e: e._check_success(), config, dict(experimental=True, training_eligible=False))
    x = LabAptExpert(env)

    def step(action):
        if len(rec.actions) > 5000:
            raise RuntimeError('5000 action limit')
        if len(rec.actions) % 100 == 0:
            print(x.phase, len(rec.actions), flush=True)
        return rec.step(action, x.phase)
    x.step = step
    r['plan'] = x.fetch()
    q, Q = x.pose()
    for _ in range(30):
        x.tick(q, Q, -1)
    r['success'] = bool(env._check_success())
except Exception as e:
    r['error'] = str(e)
    traceback.print_exc()
finally:
    signal.alarm(0)
    if x is not None:
        r.update(phase=x.phase, steps=len(rec.actions))
    if rec is not None and rec.actions:
        rec.save(a.output / 'probe.hdf5')
    if env is not None:
        env.close()
if r['success']:

    def recreate(cfg):
        e = suite.make(**cfg)
        e.reset()
        return e
    r['replay'] = validate_replay(a.output / 'probe.hdf5', recreate, snapshot, lambda e: e._check_success(), min_base_displacement=0.3, min_loaded_transport=0.3)
    r['accepted'] = bool(r['replay']['passed'])
(a.output / 'result.json').write_text(json.dumps(r, indent=2) + '\n')
print(json.dumps(r), flush=True)
