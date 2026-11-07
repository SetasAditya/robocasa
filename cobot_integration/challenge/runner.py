"""Scene freezing, private qualification and matched controller evaluation.

CLI artifacts are operator-owned. Never expose manifests, geometry qualification
or env objects to a policy. Python policy plugins are trusted evaluation code,
not a security sandbox for arbitrary agent-generated programs.
"""
import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import time

from .catalog import VERSION, PROFILES, DIFFICULTIES, MODES, plan, recipe


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, data):
    Path(path).write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False)+'\n')


def source_hash():
    root = Path(__file__).resolve().parent
    h = hashlib.sha256()
    # Include runtime sources, not just the new controller/task files.
    repo = root.parents[1]
    files = list(root.glob('*.py')) + list((repo/'robocasa').rglob('*.py')) + list((repo/'cobot_integration').glob('*.py'))
    for p in sorted(files):
        if p.is_relative_to(repo/'robocasa/models/assets'):
            continue  # Assets are bound separately by per-scene mesh/texture hashes.
        h.update(str(p.relative_to(repo)).encode()); h.update(p.read_bytes())
    return h.hexdigest()


def make(r, seed, geometry=None):
    os.environ.setdefault('MUJOCO_GL', 'egl')
    os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')
    os.environ.setdefault('PYNPUT_BACKEND', 'dummy')
    import sys
    import numpy as np
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from . import register
    register()
    from kitchen_tasks import make_env
    from cobot import controller_config
    controller = controller_config()
    controller['body_parts']['right'].update(input_type='delta', input_ref_frame='base', kp=300.)
    np.random.seed(seed)
    return make_env(r['env_name'], seed, layout=100, style=1,
                    difficulty=r['difficulty'], initialization_mode=r['mode'], operator_geometry=geometry,
                    render=False, obj_instance_split='target', controller_configs=controller,
                    robot_spawn_deviation_pos_x=0., robot_spawn_deviation_pos_y=0.)


def hold(env):
    import numpy as np
    # All methods use the same normalized base-frame delta OSC. Zero is a hold.
    return np.zeros(env.action_dim)


def apply(env, action):
    import numpy as np
    a = np.asarray(action, dtype=float).copy()
    low, high = env.action_spec
    if a.shape != low.shape or not np.isfinite(a).all() or (a < low).any() or (a > high).any():
        raise ValueError('Action outside common robot action contract')
    return env.step(a)


def render_camera(env, camera, *, depth=False, width=640, height=360):
    from robosuite.utils.binding_utils import MjRenderContextOffscreen
    from kitchen_recording import camera_render_scope
    if env.sim._render_context_offscreen is None:
        context = MjRenderContextOffscreen(env.sim, device_id=int(os.environ.get('MUJOCO_EGL_DEVICE_ID', '0')))
        if env.sim._render_context_offscreen is None:
            env.sim.add_render_context(context)
    env.has_offscreen_renderer = True
    with camera_render_scope(env, 'kinect' if camera=='mobilebase0_kinect' else None):
        result = env.sim.render(width=width, height=height, camera_name=camera, depth=depth)
    if depth:
        rgb, buffer = result
        return rgb[::-1].copy(), buffer[::-1].copy()
    return result[::-1].copy()


def bootstrap_scene(env, record, folder):
    import numpy as np
    meta = json.loads((folder/'episode.json').read_text())
    env.set_ep_meta(meta)
    try:
        if env.model is None:
            env.reset()
        env.reset_from_xml_string((folder/'model.xml').read_text())
    finally:
        env.unset_ep_meta()
    env.sim.reset()
    saved = np.load(folder/'state.npy', allow_pickle=False)
    env.sim.set_state_from_flattened(saved)
    env.sim.forward()
    if not np.array_equal(saved, env.sim.get_state().flatten()):
        raise ValueError('Frozen physical state failed exact restore')
    env.restore_challenge_state(json.loads((folder/'task_state.json').read_text()))
    controls = json.loads((folder/'controller_state.json').read_text())
    for robot, saved in zip(env.robots, controls):
        for part, value in saved.items():
            robot.gripper[part].current_action = np.asarray(value, dtype=float)
    env.timestep = 0


def load(folder):
    folder = Path(folder)
    record = json.loads((folder/'scene.json').read_text())
    if record['version'] != VERSION or record['source_sha256'] != source_hash():
        raise ValueError('Scene sources or suite version changed; rebuild in a new version')
    for file, sha in record.get('asset_files', {}).items():
        if digest(file) != sha:
            raise ValueError('Frozen scene asset changed: '+file)
    for file, sha in record['files'].items():
        if digest(folder/file) != sha:
            raise ValueError('Scene artifact changed: '+file)
    geometry = json.loads((folder/'task_state.json').read_text())['geometry']
    env = make(record['recipe'], record['seed'], geometry)
    try:
        bootstrap_scene(env, record, folder)
    except BaseException:
        env.close(); raise
    return env, record


def prepare(folder, r, seed=None, *, split='development', initialization=None, geometry=None):
    import numpy as np
    folder = Path(folder)
    reserved = plan(r['profile'], r['difficulty'], r['mode'], split, count=1)['scenes'][0]['seed']
    seed = reserved if seed is None else seed
    if not reserved <= seed < reserved+10000:
        raise ValueError('Seed is outside this task/difficulty/mode/split reserved interval')
    if folder.exists():
        raise FileExistsError('Refusing to replace frozen scene '+str(folder))
    if (r['mode']=='manipulation' or r['kind']=='recovery') and initialization is None:
        raise ValueError('This cell requires a recorded physical initialization action sequence')
    folder.mkdir(parents=True)
    env = None
    try:
        geometry_record = json.loads(Path(geometry).read_text()) if geometry else {}
        if geometry_record:
            from .goals import validate_geometry
            validate_geometry(r['kind'], geometry_record)
        env = make(r, seed, geometry_record)
        env.reset()
        init = np.load(initialization, allow_pickle=False) if initialization else np.empty((0, env.action_dim))
        if initialization and (init.ndim != 2 or len(init) == 0):
            raise ValueError('Initialization must contain physical robot actions')
        for action in init:
            apply(env, action)
        env.sim.forward()
        if r['mode']=='manipulation':
            site = env.robots[0].eef_site_id['right']
            if np.linalg.norm(env.sim.data.site_xpos[site]-env._pose('obj')[0]) > .35:
                raise ValueError('Manipulation checkpoint must begin within 35 cm of the target')
        if r['kind']=='recovery':
            if not env.check_contact(env.robots[0].gripper['right'], env.objects['obj']):
                raise ValueError('Recovery initialization has no physical gripper-object contact')
            geometry_record = getattr(env, 'qualified_geometry', {})
            nominal = geometry_record.get('nominal_grasp_tool_xyz')
            if nominal is None:
                raise ValueError('Recovery needs the reviewed nominal object position in the tool frame')
            site = env.robots[0].eef_site_id['right']
            tool_R = env.sim.data.site_xmat[site].reshape(3, 3)
            measured = tool_R.T @ (env._pose('obj')[0]-env.sim.data.site_xpos[site])
            offset = np.linalg.norm(measured-np.asarray(nominal))
            if abs(offset-r['parameters']['grasp_offset_m']) > .005:
                raise ValueError('Recorded grasp offset does not match this difficulty')
            # Require a settled, physically sustained grasp; no welded/teleported objects.
            p0 = env._pose('obj')[0].copy()
            for _ in range(20):
                apply(env, hold(env))
            if np.linalg.norm(env._pose('obj')[0]-p0) > .01:
                raise ValueError('Recovery grasp not stable for one second')
        # Temporal evaluation starts at the frozen checkpoint, preserving protected references.
        env.score_state.reset()
        env.initial_target = env._pose('obj')[0].copy()
        state = env.sim.get_state().flatten().copy()
        np.save(folder/'state.npy', state)
        np.save(folder/'initialization.npy', init)
        (folder/'model.xml').write_text(env.model.get_xml())
        write(folder/'episode.json', env.get_ep_meta())
        write(folder/'task_state.json', env.challenge_state())
        write(folder/'controller_state.json', [{part:gripper.current_action.tolist() for part,gripper in robot.gripper.items()} for robot in env.robots])
        # Record actual sampled assets and model physical parameters for audits.
        write(folder/'physics.json', dict(timestep=float(env.sim.model.opt.timestep),
              friction=env.sim.model.geom_friction.tolist(), mass=env.sim.model.body_mass.tolist(),
              action_low=env.action_spec[0].tolist(), action_high=env.action_spec[1].tolist()))
        import xml.etree.ElementTree as ET
        assets = {}
        for item in ET.fromstring((folder/'model.xml').read_text()).iter():
            if item.tag in ('mesh','texture','hfield') and item.get('file'):
                asset = Path(item.get('file')).resolve()
                assets[str(asset)] = digest(asset)
        record = dict(version=VERSION, recipe=r, seed=seed, split=split, asset_files=assets, source_sha256=source_hash(),
                      status='experimental', files={p.name:digest(p) for p in folder.iterdir() if p.is_file()})
        write(folder/'scene.json', record)
        bootstrap_scene(env, record, folder)
        return record
    except Exception as exc:
        write(folder/'PREPARATION_FAILED.json', dict(error=f'{type(exc).__name__}: {exc}'))
        raise
    finally:
        if env is not None:
            env.close()


def scene_id(folder):
    return digest(Path(folder)/'scene.json')


def replay(folder, actions):
    import numpy as np
    env, record = load(folder)
    try:
        states, scores, penetrations = [], [], []
        for action in np.asarray(actions):
            apply(env, action)
            states.append(env.sim.get_state().flatten().copy())
            scores.append(env.challenge_state())
            penetrations.append(any(c.dist < -.005 for c in env.sim.data.contact))
        return dict(success=bool(env._check_success()),
                    failure=env.score_state.failed, steps=len(states),
                    max_penetration_exceeded=any(penetrations)), np.asarray(states), scores
    finally:
        env.close()


def qualify(folder, witness=None, attestations=None):
    import numpy as np
    folder = Path(folder)
    env, record = load(folder)
    checks = {}
    evidence = dict(scene_sha256=scene_id(folder), source_sha256=source_hash())
    try:
        initial = env.sim.get_state().flatten().copy()
        task_initial = env.challenge_state()
        checks['initially_incomplete'] = not env._check_success() and not env._facts()['goal']
        checks['initial_contacts'] = not env._facts()['penetration']
        checks['initial_fixture_contacts'] = not any(c.dist < -.005 for c in env.sim.data.contact)
        checks['replay'] = np.array_equal(initial, env.sim.get_state().flatten())
        env.has_offscreen_renderer = True
        for camera in ('mobilebase0_kinect', 'robot0_eye_in_hand'):
            rgb = render_camera(env, camera, width=320, height=240)
            checks[camera] = bool(np.isfinite(rgb).all() and rgb.std()>1)
        for _ in range(40):
            apply(env, hold(env))
        checks['passive_stability'] = bool(not env._check_success() and not env.score_state.failed
                                         and np.linalg.norm(env._pose('obj')[0]-np.asarray(task_initial['initial_target'])) < .02)
        # Exact recipe geometry needs measured collision/fit validation; a reset cannot certify it.
        attest = json.loads(Path(attestations).read_text()) if attestations else {}
        checks['physical_fit_review'] = bool(attest.get('scene_sha256') == evidence['scene_sha256']
            and attest.get('collision_geometry_reviewed') is True
            and attest.get('primary_axis_verified') is True
            and attest.get('initialization_verified') is True
            and isinstance(attest.get('reviewer'), str) and attest['reviewer'].strip())
        if attestations:
            evidence['attestations'] = attest
        checks['witness'] = False
    finally:
        env.close()
    if witness:
        actions = np.load(witness, allow_pickle=False)
        result, states, scores = replay(folder, actions)
        again, states2, scores2 = replay(folder, actions)
        checks['witness'] = bool(result['success'] and not result['failure']
                                  and not result['max_penetration_exceeded']
                                  and result == again and np.allclose(states, states2, atol=1e-9, rtol=0)
                                  and scores == scores2)
        # Witness files live outside the scene identity; qualification binds their hashes.
        np.save(folder/'witness.npy', actions)
        evidence['witness_sha256'] = digest(folder/'witness.npy')
        evidence['witness_result'] = result
    evidence.update(checks=checks, qualified=all(checks.values()),
                    status='qualified' if all(checks.values()) else 'experimental')
    write(folder/'qualification.json', evidence)
    return evidence


class SensorInterface:
    """Observation boundary for trusted plugins; no task truth or success predicate."""
    def __init__(self, env):
        self.__env = env
        self.__start_base = None

    def observe(self):
        import numpy as np
        env = self.__env
        joint_ids = env.robots[0]._ref_joint_pos_indexes
        site = env.robots[0].eef_site_id['right']
        env.has_offscreen_renderer = True
        rgb, depth = render_camera(env, 'mobilebase0_kinect', depth=True)
        wrist = render_camera(env, 'robot0_eye_in_hand')
        from robosuite.utils.camera_utils import get_real_depth_map, get_camera_intrinsic_matrix, get_camera_extrinsic_matrix
        base_id = env.sim.model.body_name2id('mobilebase0_base')
        B = np.eye(4); B[:3,:3] = env.sim.data.body_xmat[base_id].reshape(3,3); B[:3,3] = env.sim.data.body_xpos[base_id]
        if self.__start_base is None:
            self.__start_base = B.copy()
        tool = np.eye(4); tool[:3,:3] = env.sim.data.site_xmat[site].reshape(3,3); tool[:3,3] = env.sim.data.site_xpos[site]
        return dict(kinect_rgb=rgb.copy(), kinect_depth_m=get_real_depth_map(env.sim, depth),
                    wrist_rgb=wrist.copy(), joint_positions=env.sim.data.qpos[joint_ids].copy(),
                    joint_velocities=env.sim.data.qvel[env.robots[0]._ref_joint_vel_indexes].copy(),
                    tool_pose_base=np.linalg.inv(B)@tool,
                    start_relative_odometry=np.linalg.inv(self.__start_base)@B,
                    base_velocity=B[:3,:3].T@env.sim.data.get_body_xvelp('mobilebase0_base'),
                    gripper_joint_positions={k:env.sim.data.qpos[v].copy() for k,v in env.robots[0]._ref_gripper_joint_pos_indexes.items()},
                    camera_intrinsics=get_camera_intrinsic_matrix(env.sim, 'mobilebase0_kinect', 360, 640),
                    camera_pose_base=np.linalg.inv(B)@get_camera_extrinsic_matrix(env.sim, 'mobilebase0_kinect'),
                    sim_time=float(env.sim.data.time))


def evaluate(folder, controller, output, *, diagnostic=False):
    """Plugin: create(instruction, action_contract)->policy; policy.act(observation)->action.

    RAX controller exports and learned policies implement the same trusted adapter.
    This does not launch RAX learning or grant its generated code filesystem access.
    """
    import numpy as np
    folder, output = Path(folder), Path(output)
    if output.exists():
        raise FileExistsError(output)
    record = json.loads((folder/'scene.json').read_text())
    if not diagnostic:
        if record.get('split') != 'formal':
            raise ValueError('Formal evaluation cannot use practice or development scenes')
        q = json.loads((folder/'qualification.json').read_text())
        if not q.get('qualified') or q['scene_sha256'] != scene_id(folder) or q['source_sha256'] != source_hash():
            raise ValueError('Formal evaluation requires current qualification and witness')
        if digest(folder/'witness.npy') != q.get('witness_sha256'):
            raise ValueError('Qualification witness changed')
    env, record = load(folder)
    output.mkdir(parents=True)
    actions = []
    metrics = dict(penetration_steps=0, dropped_object_steps=0, protected_violation_steps=0)
    started = time.perf_counter()
    result = dict(scene_sha256=scene_id(folder), diagnostic=diagnostic,
                  controller=controller, success=False)
    try:
        module = importlib.import_module(controller)
        result['controller_sha256'] = digest(module.__file__)
        contract = dict(low=env.action_spec[0].tolist(), high=env.action_spec[1].tolist(),
                        control_hz=20, arm_frame='base', arm_input='delta',
                        arm_output_min=env.robots[0].part_controllers['right'].output_min.tolist(),
                        arm_output_max=env.robots[0].part_controllers['right'].output_max.tolist(),
                        slices={k:list(v) for k,v in env.robots[0].composite_controller._action_split_indexes.items()})
        policy = module.create(instruction=record['recipe']['instruction'], action_contract=contract)
        sensors = SensorInterface(env)
        for _ in range(record['recipe']['horizon']):
            action = np.asarray(policy.act(sensors.observe()), dtype=float)
            apply(env, action); actions.append(action.copy())
            metrics['penetration_steps'] += int(any(c.dist < -.005 for c in env.sim.data.contact))
            metrics['dropped_object_steps'] += int(env._pose('obj')[0][2] < .05)
            metrics['protected_violation_steps'] += int(not env._protected())
            if env._check_success():
                result['success'] = True; break
        result['private_final_state'] = env.challenge_state()
    except Exception as exc:
        result['error'] = f'{type(exc).__name__}: {exc}'
    finally:
        result.update(steps=len(actions), wall_seconds=time.perf_counter()-started, metrics=metrics)
        np.save(output/'actions.npy', np.asarray(actions))
        write(output/'result.json', result)
        env.close()
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    q = sub.add_parser('plan')
    q.add_argument('--output', type=Path, required=True)
    for cmd in ('prepare', 'qualify', 'evaluate'):
        q = sub.add_parser(cmd); q.add_argument('--scene', type=Path, required=True)
        if cmd=='prepare':
            q.add_argument('--profile', choices=PROFILES, required=True)
            q.add_argument('--difficulty', choices=DIFFICULTIES, default='easy')
            q.add_argument('--mode', choices=MODES, default='workflow')
            q.add_argument('--seed', type=int)
            q.add_argument('--split', choices=('development','practice','formal'), default='development')
            q.add_argument('--initialization', type=Path)
            q.add_argument('--geometry', type=Path)
        elif cmd=='qualify':
            q.add_argument('--witness', type=Path); q.add_argument('--attestations', type=Path)
        else:
            q.add_argument('--controller', required=True); q.add_argument('--output', type=Path, required=True)
            q.add_argument('--diagnostic', action='store_true')
    a = p.parse_args()
    if a.command=='plan':
        if a.output.exists(): raise FileExistsError(a.output)
        write(a.output, [plan(t,d,m,s) for t in PROFILES for d in DIFFICULTIES for m in MODES for s in ('practice','formal')])
    elif a.command=='prepare':
        print(json.dumps(prepare(a.scene, recipe(a.profile,a.difficulty,a.mode),a.seed,
                                 split=a.split,initialization=a.initialization,geometry=a.geometry), indent=2))
    elif a.command=='qualify':
        print(json.dumps(qualify(a.scene,a.witness,a.attestations), indent=2))
    else:
        print(json.dumps(evaluate(a.scene,a.controller,a.output,diagnostic=a.diagnostic), indent=2))
