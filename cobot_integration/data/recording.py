"""One simulator owner; synchronized pre-action capture for native UI and experts."""
import importlib
import io
from pathlib import Path
import time
import h5py
import numpy as np
from PIL import Image
from .contract import action
from .manifest import VERSION, file_sha, read, write
from .runtime import observation, task_state, contacts, restore, sources, begin_step


class Recorder:
    def __init__(self, env, scene, output, producer):
        self.env=env;self.scene=Path(scene).resolve();self.record=read(self.scene/'scene.json')
        if self.record['initial_contacts'] or self.record['initial_success']:
            raise ValueError('Scene failed initial qualification')
        self.path=Path(output)
        if self.path.exists():raise FileExistsError(self.path)
        self.path.mkdir(parents=True)
        self.spec=self.record['contract'];self.count=0;self.started=float(env.sim.data.time)
        self.start_wall=time.monotonic();self.bad_contacts=0;self.finished=False;self.last_success=False
        self.meta=dict(version=VERSION,scene=str(self.scene),scene_id=self.record['id'],
            source_sha256=sources(),request=self.record['request'],contract=self.spec,
            instruction=self.record['instruction'],producer=producer,status='recording',training_eligible=False)
        write(self.path/'episode.json',self.meta)
        self.h5=h5py.File(self.path/'capture.h5','w')
        self.initial=env.sim.get_state().flatten().copy()
        self._dataset('sim_state',self.initial.shape,'f8')
        self._dataset('state',(18,),'f4');self._dataset('action',(self.spec['action_dim'],),'f4')
        self._dataset('timestamp',(),'f8');self._dataset('success',(),'?')
        self._dataset('contact_violation',(),'?')
        self._dataset('task_state',(),h5py.string_dtype())
        self._dataset('phase',(),h5py.string_dtype())
        for camera in ('kinect','wrist'):self._dataset(camera,(),h5py.vlen_dtype(np.dtype('uint8')))

    def _dataset(self,name,shape,dtype):
        self.h5.create_dataset(name,shape=(0,*shape),maxshape=(None,*shape),dtype=dtype,chunks=True)

    def _append(self,name,value):
        ds=self.h5[name];ds.resize(self.count+1,axis=0);ds[self.count]=value

    def observe(self):
        return {**observation(self.env),'task':self.record['instruction']}

    def step(self,command,phase=''):
        if self.finished:raise RuntimeError('Episode already closed')
        a=action(command,self.spec)
        obs=self.observe()
        begin_step(self.env)
        self._append('sim_state',self.env.sim.get_state().flatten().copy())
        self._append('state',obs['observation.state'])
        self._append('timestamp',float(self.env.sim.data.time)-self.started)
        self._append('action',a.astype(np.float32))
        import json
        self._append('task_state',json.dumps(task_state(self.env),sort_keys=True))
        self._append('phase',str(phase))
        for name in ('kinect','wrist'):
            buffer=io.BytesIO();Image.fromarray(obs['observation.images.'+name]).save(buffer,format='JPEG',quality=90)
            self._append(name,np.frombuffer(buffer.getvalue(),dtype='uint8'))
        # Execute exactly the float32 values that will be read back from the dataset.
        result=self.env.step(a.astype(np.float32).astype(np.float64).copy())
        violation=bool(contacts(self.env));self.bad_contacts+=int(violation)
        self._append('contact_violation',violation);self.last_success=bool(self.env._check_success());self._append('success',self.last_success)
        self.count+=1
        if self.count%20==0:self.h5.flush()
        return result

    def close(self,reason='completed'):
        if self.finished:return self.meta
        self.finished=True
        self.h5.create_dataset('final_state',data=self.env.sim.get_state().flatten())
        self.h5.flush();self.h5.close()
        self.meta.update(status='recorded' if reason=='completed' else reason,frames=self.count,
            success=self.last_success,contact_violation_steps=self.bad_contacts,
            wall_seconds=time.monotonic()-self.start_wall,capture_sha256=file_sha(self.path/'capture.h5'))
        write(self.path/'episode.json',self.meta,overwrite=True)
        return self.meta


class Context:
    def __init__(self,recorder,max_steps,privileged=False):
        self._recorder=recorder;self.max_steps=max_steps;self.privileged=privileged
        self.instruction=recorder.record['instruction'];self.action_contract=recorder.spec
    @property
    def env(self):
        if not self.privileged:raise PermissionError('Simulator truth is only available to explicitly privileged teachers')
        return self._recorder.env
    def observe(self):return self._recorder.observe()
    @property
    def success(self):return self._recorder.last_success
    def step(self,command,phase=''):
        if self._recorder.count>=self.max_steps:raise RuntimeError('Episode step budget exhausted')
        return self._recorder.step(command,phase)
    def action_vector(self,parts):
        return self._recorder.env.robots[0].create_action_vector(parts).copy()


def collect(scene,output,expert,max_steps=1800,privileged=False):
    env,record=restore(scene);rec=None
    try:
        module=importlib.import_module(expert)
        rec=Recorder(env,scene,output,dict(kind='expert_plugin',module=expert,sha256=file_sha(module.__file__),privileged=privileged))
        (rec.path/'expert_source.py').write_bytes(Path(module.__file__).read_bytes())
        module.run(Context(rec,max_steps,privileged))
        return rec.close()
    except BaseException:
        if rec is not None:rec.close('interrupted_or_failed')
        raise
    finally:env.close()


def teleop(scene,output,device_name='keyboard',max_steps=1800):
    env,record=restore(scene,gui=True);rec=None
    try:
        from robosuite.devices import Keyboard,SpaceMouse
        from robocasa.scripts.collect_demos import collect_human_trajectory
        if device_name=='keyboard':device=Keyboard(env=env,pos_sensitivity=4.,rot_sensitivity=4.)
        else:
            import robocasa.macros as macros
            device=SpaceMouse(env=env,pos_sensitivity=4.,rot_sensitivity=4.,vendor_id=macros.SPACEMOUSE_VENDOR_ID,product_id=macros.SPACEMOUSE_PRODUCT_ID)
        from cobot import frame_viewer
        frame_viewer(env)
        rec=Recorder(env,scene,output,dict(kind='native_teleop',device=device_name))
        env.ep_directory=str(output)
        _,discard=collect_human_trajectory(env,device,'right','single-arm-opposed',mirror_actions=True,
            render=True,max_fr=20,reset_fn=lambda:None,step_fn=rec.step,warmup_steps=0,
            close_env=False,max_steps=max_steps,success_fn=lambda:rec.last_success)
        return rec.close('discarded' if discard else 'completed')
    except BaseException:
        if rec is not None:rec.close('interrupted_or_failed')
        raise
    finally:env.close()
