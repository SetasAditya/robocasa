"""Exact native command contract; measured state and commands have distinct semantics."""
import numpy as np
from .manifest import VERSION, identity


def describe(env):
    robot=env.robots[0];ctrl=robot.part_controllers['right']
    if ctrl.input_type!='delta' or ctrl.input_ref_frame!='base' or env.control_freq!=20:
        raise ValueError('Expected 20 Hz base-frame delta OSC')
    slices={k:[int(a),int(b)] for k,(a,b) in robot.composite_controller._action_split_indexes.items()}
    names=[]
    for part,(a,b) in sorted(slices.items(),key=lambda item:item[1][0]):
        if a!=len(names):raise ValueError('Noncontiguous native action slices')
        names.extend(f'{part}.command_{i}' for i in range(b-a))
    low,high=env.action_spec
    if len(names)+1==env.action_dim:
        # HybridMobileBase appends its mode outside _action_split_indexes.
        negative=robot.create_action_vector({'base_mode':-1})
        positive=robot.create_action_vector({'base_mode':1})
        changed=np.flatnonzero(positive!=negative)
        if changed.tolist()!=[len(names)] or negative[-1]!=-1 or positive[-1]!=1:
            raise ValueError('Unknown composite action extension')
        slices['base_mode']=[len(names),len(names)+1];names.append('base_mode.command_0')
    if len(names)!=env.action_dim:raise ValueError('Incomplete native action layout')
    base=robot.part_controllers.get('base')
    base_description=dict(controller=type(base).__name__,command='native normalized command',mode='positive: desired-goal tracking; nonpositive: achieved-goal tracking')
    for key in ('input_min','input_max','output_min','output_max'):
        if hasattr(base,key):base_description[key]=np.asarray(getattr(base,key)).tolist()
    spec=dict(version=VERSION,fps=20,robot='CobotKinovaFourBar',state_dim=18,action_dim=env.action_dim,
              action_names=names,slices=slices,low=low.tolist(),high=high.tolist(),
              arm=dict(type='OSC_POSE',frame='base',input='delta',kp=np.asarray(ctrl.kp).tolist(),
                       output_min=ctrl.output_min.tolist(),output_max=ctrl.output_max.tolist()),
              gripper='native command, distinct from measured closedness',
              base=base_description,
              cameras={'kinect':[360,640,3],'wrist':[360,640,3]},
              alignment='observation before action; simulation clock',
              solver_boundary='zero qacc_warmstart and refresh OSC before sensors and again before each action',
              privileged_policy_inputs=False)
    spec['hash']=identity(spec)
    return spec


def action(value, spec):
    a=np.array(value,dtype=np.float64,copy=True)
    if a.shape!=(spec['action_dim'],) or not np.isfinite(a).all():raise ValueError('Invalid native action')
    if np.any(a<np.asarray(spec['low'])) or np.any(a>np.asarray(spec['high'])):
        raise ValueError('Action outside controller bounds')
    return a


def features(spec):
    return {
      'observation.images.kinect':dict(dtype='video',shape=(360,640,3),names=['height','width','channels']),
      'observation.images.wrist':dict(dtype='video',shape=(360,640,3),names=['height','width','channels']),
      'observation.state':dict(dtype='float32',shape=(18,),names=[*[f'joint_{i}.position_rad' for i in range(1,8)],*[f'joint_{i}.velocity_rad_s' for i in range(1,8)],'gripper.measured_closedness','base.vx_m_s','base.vy_m_s','base.wz_rad_s']),
      'action':dict(dtype='float32',shape=(spec['action_dim'],),names=spec['action_names'])}
