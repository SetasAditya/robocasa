"""Isolated action-only controller probes; never accepted training data."""
import os
os.environ.setdefault('MUJOCO_GL', 'egl')
import argparse, json, signal, time, traceback
from pathlib import Path
from kitchen_tasks import environment_config
from moma_validation import EpisodeRecorder, validate_replay
import cobot_fourbar

def run(task, seed, out):
    out.mkdir(parents=True, exist_ok=False)
    env = None
    rec = None
    x = None
    start = time.time()
    from . import snapshot_sources
    snapshot_sources(out)
    result = dict(task=task, seed=seed, layout=100, robot='CobotKinovaFourBar', controller='existing experimental physical primitives; privileged fixture poses', training_eligible=False, success=False, replay_passed=False)

    def deadline(*_):
        raise TimeoutError('Probe exceeded 240 seconds')
    signal.signal(signal.SIGALRM, deadline)
    signal.alarm(240)
    try:
        import robosuite as suite
        config = environment_config(task, seed, 100, 1, render=False, images=False, robots='CobotKinovaFourBar')
        env = suite.make(**config)
        env.reset()
        env.sim.forward()
        fixture = env.drawer if task == 'OpenDrawer' else env.fxtr
        result['fixture'] = fixture.name
        result['episode_metadata'] = env.get_ep_meta()
        result['initial_success'] = bool(env._check_success())
        if result['initial_success']:
            raise RuntimeError('Task already successful at reset')
        result['initial_penetrations_over_5mm'] = [dict(depth_m=float(-c.dist), geoms=[env.sim.model.geom_id2name(int(g)) for g in (c.geom1, c.geom2)]) for c in env.sim.data.contact if c.dist < -0.005]
        if result['initial_penetrations_over_5mm']:
            raise RuntimeError('Initial penetration exceeds 5 mm')
        handles = [g.get('name') for g in fixture.get_obj().iter('geom') if 'handle' in g.get('name', '') and g.get('group', '0') == '0']
        joints = [j.get('name') for j in fixture.get_obj().iter('joint')]
        result.update(handles=handles, joints=joints, initial_joint_positions={j: float(env.sim.data.get_joint_qpos(j)) for j in joints})

        def snapshot(e):
            g = e.robots[0].gripper['right']
            groups = [g.important_geoms[k] for k in ('left_finger', 'right_finger')]
            return dict(base_xy=e.sim.data.get_body_xpos('mobilebase0_base')[:2].copy(), object_xyz=e.sim.data.get_geom_xpos(handles[0]).copy(), object_name='fixture_handle', grasped=bool(e._check_grasp(groups, handles)))
        rec = EpisodeRecorder(env, snapshot, lambda e: e._check_success(), config, dict(probe=True, training_eligible=False))
        from .experts import LabAptExpert
        from .cabinet import LabAptCabinetExpert
        x = LabAptCabinetExpert(env) if task == 'OpenCabinet' else LabAptExpert(env)

        def step(action):
            if len(rec.actions) >= 6000:
                raise RuntimeError('6000-action probe limit')
            if len(rec.actions) % 100 == 0:
                print(json.dumps(dict(task=task, seed=seed, step=len(rec.actions), phase=x.phase)), flush=True)
            return rec.step(action, x.phase)
        x.step = step
        plans = []
        for body in fixture.get_obj().iter('body'):
            joint = body.find('joint')
            if joint is None:
                continue
            j = joint.get('name')
            if task == 'OpenFridge' and 'freezer' in j:
                continue
            hs = [g.get('name') for g in body.iter('geom') if g.get('name') in handles]
            if hs:
                plans.append((j, hs[0]))
        result['adapted_controller'] = True
        result['hinge_plan'] = plans
        for j, h in plans:
            x.open_fixture(fixture, j, h)
        p, Q = x.pose()
        x.phase = 'final_hold'
        for _ in range(30):
            x.tick(p, Q, -1)
        result['success'] = bool(env._check_success())
    except Exception as exc:
        result['error'] = type(exc).__name__ + ': ' + str(exc)
        result['traceback'] = traceback.format_exc()
    finally:
        signal.alarm(0)
        if x is not None:
            result.update(phase=x.phase, steps=len(rec.actions))
        if env is not None:
            try:
                result['final_native_success'] = bool(env._check_success())
                result['final_joint_positions'] = {j: float(env.sim.data.get_joint_qpos(j)) for j in result.get('joints', [])}
            except Exception:
                pass
        if rec is not None and rec.actions:
            rec.save(out / 'probe.hdf5')
        if env is not None:
            env.close()
    if result['success']:

        def recreate(cfg):
            e = suite.make(**cfg)
            e.reset()
            return e
        report = validate_replay(out / 'probe.hdf5', recreate, snapshot, lambda e: e._check_success(), min_base_displacement=0, min_loaded_transport=0)
        result['replay_passed'] = bool(report['passed'])
    result['seconds'] = time.time() - start
    (out / 'result.json').write_text(json.dumps(result, indent=2, default=lambda x: x.tolist() if hasattr(x, 'tolist') else str(x)) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ('episode_metadata', 'traceback')}, default=str), flush=True)
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--task', required=True)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    run(a.task, a.seed, a.output)
