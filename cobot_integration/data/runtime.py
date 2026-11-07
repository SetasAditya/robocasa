"""Freeze and restore source MJCF, physical state, task history and gripper commands."""
import os
from pathlib import Path
import sys
import xml.etree.ElementTree as ET
import numpy as np
from .manifest import VERSION, read, write, identity, file_sha, check_request
from .contract import describe


def bootstrap(gui=False):
    if not gui:
        os.environ.setdefault('MUJOCO_GL','egl');os.environ.setdefault('PYOPENGL_PLATFORM','egl')
    if not gui:os.environ.setdefault('PYNPUT_BACKEND','dummy')
    integration=Path(__file__).resolve().parents[1]
    if str(integration) not in sys.path:sys.path.insert(0,str(integration))
    import robocasa,cobot_fourbar  # noqa: F401
    from ..challenge import register
    register()


def sources():
    from ..challenge.runner import source_hash
    return identity(dict(runtime=source_hash(),data={p.name:file_sha(p) for p in Path(__file__).parent.glob('*.py')}))


def make(request, *, geometry=None, gui=False):
    check_request(request);bootstrap(gui)
    from kitchen_tasks import make_env
    from cobot import controller_config
    cfg=controller_config()
    cfg['body_parts']['right'].update(input_type='delta',input_ref_frame='base',kp=300.)
    extras={}
    name=request['task']
    if request['track']=='challenge':
        from ..challenge.catalog import recipe
        r=recipe(name,request['difficulty'],request['mode']);name=r['env_name']
        extras.update(difficulty=request['difficulty'],initialization_mode=request['mode'],operator_geometry=geometry or {})
    np.random.seed(request['seed'])
    return make_env(name,request['seed'],layout=request['layout'],style=request['style'],
        controller_configs=cfg,render=False,has_renderer=gui,renderer='mjviewer',
        use_object_obs=False,obj_instance_split=request['asset_split'],
        robot_spawn_deviation_pos_x=0.,robot_spawn_deviation_pos_y=0.,**extras)


def task_state(env):
    from .task_state import capture
    return capture(env)


def control_state(env):
    from .task_state import pack
    names=('goal_pos','goal_ori','_goal_update_mode','initial_joint','init_pos','init_ori','goal_qvel','last_goal_qvel','goal_vel')
    return [dict(grippers={key:g.current_action.tolist() for key,g in r.gripper.items()},
        controllers={part:{key:pack(getattr(c,key)) for key in names if hasattr(c,key)} for part,c in r.part_controllers.items()}) for r in env.robots]


def restore_python(env, task, controls):
    from .task_state import restore,unpack
    restore(env,task)
    for robot,saved in zip(env.robots,controls):
        for key,values in saved['grippers'].items():robot.gripper[key].current_action=np.asarray(values,dtype=float)
        for part,values in saved['controllers'].items():
            for key,value in values.items():setattr(robot.part_controllers[part],key,unpack(value))
    env.timestep=0


def canonicalize(env,xml,meta,state,task,controls):
    env.set_ep_meta(meta)
    try:
        if env.model is None:env.reset()
        env.reset_from_xml_string(xml)
    finally:env.unset_ep_meta()
    env.sim.reset();env.sim.set_state_from_flattened(state);env.sim.forward()
    if not np.array_equal(state,env.sim.get_state().flatten()):raise ValueError('Physical restore mismatch')
    restore_python(env,task,controls)
    env.update_state()


def prepare(request, output, *, geometry=None, initialization=None):
    output=Path(output)
    if output.exists():raise FileExistsError(output)
    check_request(request)
    if (request['mode']=='manipulation' or request['task']=='ImperfectGraspRecovery') and not initialization:
        raise ValueError('Recorded physical initialization actions are required')
    if request['track']=='challenge':
        from ..challenge.catalog import recipe
        from ..challenge.goals import validate_geometry
        kind=recipe(request['task'])['kind']
        if kind in ('rack','plate','stack','nest','recovery') and not geometry:
            raise ValueError('Reviewed task geometry configuration is required')
        if geometry:validate_geometry(kind,geometry)
    output.mkdir(parents=True)
    env=None
    try:
        env=make(request,geometry=geometry);env.reset()
        spec=describe(env)
        if initialization:
            from .contract import action
            actions=np.load(initialization,allow_pickle=False)
            if actions.ndim!=2 or len(actions)==0:raise ValueError('Initialization must be an action sequence')
            for a in actions:env.step(action(a,spec))
            np.save(output/'initialization.npy',actions)
        if request['mode']=='manipulation':
            point=env.sim.data.site_xpos[env.robots[0].eef_site_id['right']]
            if np.linalg.norm(point-env._pose('obj')[0])>.35:raise ValueError('Approach checkpoint is too far from target')
        if request['task']=='ImperfectGraspRecovery':
            if not env.check_contact(env.robots[0].gripper['right'],env.objects['obj']):raise ValueError('No physical recovery grasp')
            sid=env.robots[0].eef_site_id['right'];R=env.sim.data.site_xmat[sid].reshape(3,3)
            measured=R.T@(env._pose('obj')[0]-env.sim.data.site_xpos[sid])
            from ..challenge.catalog import recipe
            offset=recipe(request['task'],request['difficulty'])['parameters']['grasp_offset_m']
            if abs(np.linalg.norm(measured-np.asarray(geometry['nominal_grasp_tool_xyz']))-offset)>.005:
                raise ValueError('Recovery offset does not match recipe')
            initial=env._pose('obj')[0].copy()
            for _ in range(20):env.step(np.zeros(env.action_dim))
            if np.linalg.norm(env._pose('obj')[0]-initial)>.01:raise ValueError('Unstable physical recovery grasp')
        if hasattr(env,'score_state'):
            env.score_state.reset()
            env.protected_initial={n:env._pose(n) for n in env.objects if n.startswith('protected_')}
            env.initial_target=env._pose('obj')[0].copy()
        initial_success=bool(env._check_success())
        if initial_success:raise ValueError('Initially completed task')
        from kitchen_recording import add_wrist_camera
        add_wrist_camera(env)
        xml=env.model.get_xml();meta=env.get_ep_meta();state=env.sim.get_state().flatten().copy()
        task=task_state(env);controls=control_state(env)
        canonicalize(env,xml,meta,state,task,controls)
        (output/'model.xml').write_text(xml);np.save(output/'state.npy',state)
        write(output/'episode.json',meta);write(output/'task.json',task);write(output/'controls.json',controls)
        assets={}
        for e in ET.fromstring(xml).iter():
            if e.tag in ('mesh','texture','hfield') and e.get('file'):
                path=Path(e.get('file')).resolve();assets[str(path)]=file_sha(path)
        record=dict(version=VERSION,request=request,source_sha256=sources(),contract=spec,
                    instruction=meta['lang'],geometry=geometry or {},assets=assets,
                    files={p.name:file_sha(p) for p in output.iterdir() if p.is_file()},
                    initial_contacts=contacts(env),initial_success=initial_success)
        record['id']=identity(record);write(output/'scene.json',record)
        return record
    except BaseException as e:
        write(output/'FAILED.json',dict(error=f'{type(e).__name__}: {e}'));raise
    finally:
        if env is not None:env.close()


def verify_scene(folder):
    folder=Path(folder);r=read(folder/'scene.json')
    check_request(r['request'])
    if r['version']!=VERSION or identity({k:v for k,v in r.items() if k!='id'})!=r['id']:
        raise ValueError('Scene identity changed')
    if r['source_sha256']!=sources():raise ValueError('Scene source revision changed')
    for name,sha in r['files'].items():
        if file_sha(folder/name)!=sha:raise ValueError('Scene artifact changed: '+name)
    for name,sha in r['assets'].items():
        if file_sha(name)!=sha:raise ValueError('Scene asset changed: '+name)
    return r


def restore(folder,gui=False):
    folder=Path(folder);r=verify_scene(folder);env=make(r['request'],geometry=r['geometry'],gui=gui)
    try:
        canonicalize(env,(folder/'model.xml').read_text(),read(folder/'episode.json'),np.load(folder/'state.npy',allow_pickle=False),read(folder/'task.json'),read(folder/'controls.json'))
        if describe(env)!=r['contract']:raise ValueError('Controller contract changed')
        return env,r
    except BaseException:env.close();raise


def contacts(env):
    from cobot_scenarios.runtime import collisions
    return collisions(env)


def observation(env):
    begin_step(env)  # Current kinematics for both cameras and measured body twist.
    from ..challenge.runner import render_camera
    from kitchen_recording import render_camera as render_collection,measured_state
    # Initialize the shared EGL context and preserve the existing optical aperture handling.
    render_camera(env,'mobilebase0_kinect',width=32,height=18)
    return {'observation.images.kinect':render_collection(env,'kinect'),
            'observation.images.wrist':render_collection(env,'wrist'),
            'observation.state':measured_state(env)}


def begin_step(env):
    """Canonical solver/cache boundary, independent of how often a teacher observes.

    This clears only MuJoCo's numerical initial guess, never qpos/qvel or task
    state. It is shared by capture, replay and learned-policy execution.
    """
    env.sim.data.qacc_warmstart[:]=0.
    env.robots[0].part_controllers['right'].update(force=True)
