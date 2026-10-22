"""Native HDF5 recording and independent action-only mobile-task replay.

A recorder never changes simulator state. Replay restores the initial state once,
then uses env.step exclusively. Snapshot callbacks must measure physical contact,
not a desired gripper command. States are pre-action; final_state is post-action.
"""
from pathlib import Path
import json
import platform
import h5py
import numpy as np


def _json(value):
    return json.dumps(value, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else str(x))


def mobility_metrics(samples):
    """Require actual base displacement and contiguous physically grasped transport."""
    if not samples:
        return dict(base_displacement=0., loaded_transport=0., finite=False)
    base=np.asarray([s['base_xy'] for s in samples], dtype=float)
    obj=np.asarray([s['object_xyz'] for s in samples], dtype=float)
    maximum=0.; start=None; identity=None; per_object={}
    for i,s in enumerate(samples):
        if bool(s['grasped']):
            if start is None or s.get('object_name') != identity:
                start=i; identity=s.get('object_name')
            distance=float(np.linalg.norm(obj[i,:2]-obj[start,:2]))
            maximum=max(maximum,distance)
            key=str(identity)
            per_object[key]=max(per_object.get(key,0.),distance)
        else: start=None
    return dict(base_displacement=float(np.linalg.norm(base-base[0],axis=1).max()),
                loaded_transport=maximum, loaded_transport_by_object=per_object, finite=bool(np.isfinite(base).all() and np.isfinite(obj).all()))


class EpisodeRecorder:
    def __init__(self, env, snapshot, success, env_config, task_metadata=None):
        self.env=env; self.snapshot=snapshot; self.success=success
        self.env_config=env_config; self.task_metadata=task_metadata or {}
        # Canonicalize exactly once, before any recorded action, using source MJCF.
        # MuJoCo's compiled-model XML serializes inertias / floats differently and
        # can diverge under contact if recompiled only on the replay side.
        self.xml=env.model.get_xml()
        self.ep_meta=env.get_ep_meta() if hasattr(env,'get_ep_meta') else {}
        self.initial_state=np.asarray(env.sim.get_state().flatten()).copy()
        if hasattr(env,'set_ep_meta'): env.set_ep_meta(self.ep_meta)
        env.reset_from_xml_string(self.xml)
        env.sim.reset()
        env.sim.set_state_from_flattened(self.initial_state)
        env.sim.forward()
        if hasattr(env,'initialize_sequence_history'):env.initialize_sequence_history()
        self.states=[]; self.actions=[]; self.samples=[snapshot(env)]; self.successes=[]; self.phases=[]
    def step(self, action, phase=''):
        action=np.asarray(action,dtype=float).copy()
        state=np.asarray(self.env.sim.get_state().flatten()).copy()
        if not np.isfinite(action).all() or not np.isfinite(state).all():
            raise ValueError('Nonfinite action or simulator state')
        # Base controllers mutate action slices in place: preserve pristine input.
        self.states.append(state); self.actions.append(action.copy()); self.phases.append(phase)
        result=self.env.step(action)
        self.samples.append(self.snapshot(self.env)); self.successes.append(bool(self.success(self.env)))
        if not np.isfinite(self.env.sim.get_state().flatten()).all():
            raise ValueError('Nonfinite post-action simulator state')
        return result
    def save(self,path):
        path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
        if not self.actions: raise ValueError('Empty episode')
        with h5py.File(path,'w') as f:
            data=f.create_group('data'); data.attrs['env_info']=_json(self.env_config)
            cfg=dict(self.env_config); name=cfg.pop('env_name')
            try:
                import robosuite
                version=robosuite.__version__
            except ImportError: version='unknown'
            data.attrs['env_args']=_json(dict(type=1,env_name=name,env_version=version,env_kwargs=cfg))
            data.attrs['total']=len(self.actions)
            data.attrs['provenance']=_json(dict(python=platform.python_version(),robosuite=version))
            g=data.create_group('demo_0');g.attrs['model_file']=self.xml;g.attrs['ep_meta']=_json(self.ep_meta)
            g.attrs['task_metadata']=_json(self.task_metadata);g.attrs['num_samples']=len(self.actions)
            g.attrs['success']=False;g.attrs['action_replay_validated']=False
            for key,value in [('states',self.states),('actions',self.actions),('final_state',self.env.sim.get_state().flatten()),('successes',self.successes)]:
                g.create_dataset(key,data=np.asarray(value),compression='gzip')
            g.create_dataset('phases',data=np.asarray(self.phases,dtype=h5py.string_dtype()))
            aux=g.create_group('physical_metrics')
            for key in ('base_xy','object_xyz','grasped'):
                aux.create_dataset(key,data=np.asarray([s[key] for s in self.samples]),compression='gzip')
            aux.create_dataset('object_name',data=np.asarray([s.get('object_name','') for s in self.samples],dtype=h5py.string_dtype()))
        return path


def validate_replay(path, make_env, snapshot, success, state_tolerance=1e-5,
                    min_base_displacement=.3,min_loaded_transport=.3,success_hold=10,
                    report_path=None):
    """make_env(config) must instantiate the same registered robot/task/controller.

    snapshot(env) returns base_xy, object_xyz, grasped (physical contacts).
    No later recorded states are injected, including during task predicates.
    """
    path=Path(path)
    report=dict(dataset=str(path),passed=False,initial_state_restored_once=True,
                subsequent_state_forcing=False)
    env=None
    try:
        with h5py.File(path,'r') as f:
            data=f['data'];g=data['demo_0'];config=json.loads(data.attrs['env_info'])
            states=g['states'][:];actions=g['actions'][:];final=g['final_state'][:]
            xml=g.attrs['model_file'];meta=json.loads(g.attrs['ep_meta']);task_metadata=json.loads(g.attrs.get('task_metadata','{}'))
        if len(states)!=len(actions) or len(actions)==0: raise ValueError('State/action length mismatch')
        if not all(np.isfinite(x).all() for x in (states,actions,final)): raise ValueError('Nonfinite recording')
        env=make_env(config)
        if hasattr(env,'set_ep_meta'): env.set_ep_meta(meta)
        if env.model is None or env.sim is None:env.reset()
        env.reset_from_xml_string(env.edit_model_xml(xml) if hasattr(env,'edit_model_xml') else xml)
        env.sim.reset();env.sim.set_state_from_flattened(states[0]);env.sim.forward()
        if hasattr(env,'initialize_sequence_history'):env.initialize_sequence_history()
        samples=[snapshot(env)]; successes=[]; max_error=0.; first_divergence=None; max_error_step=None; max_error_component=None
        for step_index,(state,action) in enumerate(zip(states,actions)):
            current=env.sim.get_state().flatten()
            if not np.isfinite(current).all(): raise ValueError('Nonfinite replay state')
            difference=np.abs(current-state)
            error=float(difference.max())
            if first_divergence is None and error>state_tolerance:
                first_divergence=dict(step=step_index,error=error,state_component=int(difference.argmax()))
            if error>max_error:
                max_error=error;max_error_step=step_index;max_error_component=int(difference.argmax())
            env.step(action.copy());samples.append(snapshot(env));successes.append(bool(success(env)))
        current=env.sim.get_state().flatten()
        if not np.isfinite(current).all(): raise ValueError('Nonfinite final replay state')
        max_error=max(max_error,float(np.max(np.abs(current-final))))
        metrics=mobility_metrics(samples)
        assigned=task_metadata.get('required_transport_objects',meta.get('required_transport_objects',meta.get('table_setting_items')))
        if assigned is None:assigned=[env.objects[a['object_index']].name for a in getattr(env,'assignments',[])]
        if not isinstance(assigned,list) or any(not isinstance(name,str) for name in assigned):raise ValueError('Invalid required_transport_objects metadata')
        available=set(env.objects) if isinstance(env.objects,dict) else {obj.name for obj in env.objects}
        if not set(assigned).issubset(available):raise ValueError('Required transport objects absent from environment')
        every_object_carried=all(metrics['loaded_transport_by_object'].get(name,0.)>=min_loaded_transport for name in assigned)
        distractor_drift=0.
        task_spec=getattr(env,'task_spec',{})
        distractors=task_spec.get('distractors',[])
        for offset,source in enumerate(distractors):
            index=len(getattr(env,'assignments',[]))+offset
            pos=env.sim.data.body_xpos[env.object_body_ids[index]]
            distractor_drift=max(distractor_drift,float(np.linalg.norm(pos[:2]-np.asarray(source)[:2])))
        held=len(successes)>=success_hold and all(successes[-success_hold:])
        report.update(metrics,max_recorded_state_error=max_error,replayed_actions=len(actions),
                      task_success_last_steps=held,success_hold=success_hold,
                      first_divergence=first_divergence,max_error_step=max_error_step,max_error_component=max_error_component,
                      required_transport_objects=assigned,every_assigned_object_carried=every_object_carried,
                      maximum_distractor_final_xy_drift=distractor_drift)
        report['passed']=bool(metrics['finite'] and held and max_error<=state_tolerance and
                              metrics['base_displacement']>=min_base_displacement and
                              metrics['loaded_transport']>=min_loaded_transport and every_object_carried and distractor_drift<=.05)
    except Exception as exc:
        report['error']=f'{type(exc).__name__}: {exc}'
    finally:
        if env is not None: env.close()
    with h5py.File(path,'a') as f:
        g=f['data/demo_0'];g.attrs['success']=report['passed'];g.attrs['action_replay_validated']=report['passed']
        g.attrs['replay_validation']=_json(report)
    Path(report_path or path.with_suffix('.validation.json')).write_text(json.dumps(report,indent=2))
    return report


def task_snapshot(env):
    """Standard MoMaSuite physical snapshot; choose currently grasped object."""
    robot=env.robots[0]
    objects=list(env.objects)
    selected=objects[0]
    grasped=False
    for obj in objects:
        if env._check_grasp(robot.gripper['right'],obj):
            selected=obj; grasped=True; break
    index=objects.index(selected)
    body_id=env.object_body_ids.get(index,env.object_body_ids.get(selected.name))
    return dict(base_xy=env.sim.data.get_body_xpos('mobilebase0_base')[:2].copy(),
                object_xyz=env.sim.data.body_xpos[body_id].copy(),grasped=bool(grasped),
                object_name=selected.name)
