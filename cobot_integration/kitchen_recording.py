"""Synchronized simulated Kinect/wrist recording; mounts are NOT real calibration."""
from pathlib import Path
import json
import os
import shutil
import io
from contextlib import contextmanager
from PIL import Image
import xml.etree.ElementTree as ET
import h5py
import numpy as np
from moma_validation import EpisodeRecorder

SOURCE=str(Path(__file__).resolve().parent/'assets/cobot_fourbar85.xml')
CAMERAS={'kinect':dict(name='mobilebase0_kinect',width=640,height=360,native_resolution=[1280,720]),
         'wrist':dict(name='robot0_collection_wrist',width=640,height=360,render_width=480,render_height=360,native_resolution=[640,480],letterbox_left_right=[80,80])}


def add_wrist_camera(env):
    body=env.model.worldbody.find(".//body[@name='robot0_Bracelet_Link']")
    if body is None:raise ValueError('Expected physical Kinova Bracelet_Link camera mount missing')
    if body.find("camera[@name='robot0_collection_wrist']") is None:
        ET.SubElement(body,'camera',name='robot0_collection_wrist',pos='0 -0.05639 -0.058475',
                      quat='0 0 0 1',fovy='41.83792730009236',resolution='640 480')


def configure_rendering(env,hide_sites=False):
    env._collection_hide_sites=bool(hide_sites)
    context=env.sim._render_context_offscreen
    if context is not None:
        if hide_sites:context.vopt.sitegroup[:]=0
        else:
            import mujoco
            context.vopt.sitegroup[:]=mujoco.MjvOption().sitegroup


KINECT_HOUSING_GEOM='mobilebase0_kinect_housing_visual'


def housing_geom_id(env):
    try:
        gid=env.sim.model.geom_name2id(KINECT_HOUSING_GEOM)
        return int(gid) if gid>=0 else None
    except (ValueError,KeyError):return None


@contextmanager
def camera_render_scope(env,key):
    """Expose the sourceopticalaperture only to its own camera, never to physics."""
    gid=housing_geom_id(env) if key=='kinect' else None
    if gid is None:
        yield
        return
    rgba=env.sim.model.geom_rgba[gid].copy()
    try:
        env.sim.model.geom_rgba[gid,3]=0.
        yield
    finally:
        env.sim.model.geom_rgba[gid]=rgba


def render_camera_raw(env,key,depth=False,segmentation=False):
    """Unflipped camera render; RGB/depth/segmentation share the same aperture."""
    configure_rendering(env,getattr(env,'_collection_hide_sites',False))
    spec=key if isinstance(key,dict) else CAMERAS[key]
    scope_key='kinect' if spec['name']==CAMERAS['kinect']['name'] else None
    with camera_render_scope(env,scope_key):
        return env.sim.render(spec.get('render_width',spec['width']),spec.get('render_height',spec['height']),camera_name=spec['name'],depth=depth,segmentation=segmentation)


def camera_metadata(env,key):
    spec=CAMERAS[key];cid=env.sim.model.camera_name2id(spec['name'])
    w,h=spec['width'],spec['height'];fovy=float(env.sim.model.cam_fovy[cid])
    fy=.5*h/np.tan(np.deg2rad(fovy)/2)
    metadata=dict(**spec,fovy_degrees=fovy,intrinsics=[[fy,0,w/2],[0,fy,h/2],[0,0,1]],
                source_model=SOURCE,calibration_status='provisional_simulation_mount_not_real_robot_calibration',
                camera_frame='MuJoCo OpenGL camera: +x right,+y up,-z forward',
                stored_image='RGB uint8 top-left origin',world_extrinsics='camera_to_world rotation3x3 and translation3',
                rendering_options=dict(site_groups_hidden=bool(getattr(env,'_collection_hide_sites',False))),
                mount_body=env.sim.model.body_id2name(env.sim.model.cam_bodyid[cid]),
                local_position=env.sim.model.cam_pos[cid].tolist(),local_quaternion_wxyz=env.sim.model.cam_quat[cid].tolist())
    if housing_geom_id(env) is not None:
        metadata['rendering_options']['optical_aperture_correction']=dict(version='source-kinect-housing-aperture-v2',housing_visual_geom=KINECT_HOUSING_GEOM,suppressed_geom_names=([KINECT_HOUSING_GEOM] if key=='kinect' else []),scope='render-only own opaque housing exclusion; wrist retains full housing',source_camera_pose_and_intrinsics_unchanged=True,all_source_chassis_triangles_preserved=True)
    return metadata


def render_camera(env,key,depth=False):
    configure_rendering(env,getattr(env,'_collection_hide_sites',False))
    spec=CAMERAS[key]
    output=render_camera_raw(env,key,depth=depth)
    if key=='wrist':
        if depth:raise ValueError('Wrist depth not supported')
        output=np.pad(output,((0,0),(80,80),(0,0)),mode='constant')
    if not depth:return output[::-1].copy()
    rgb,buffer=output;near=env.sim.model.vis.map.znear*env.sim.model.stat.extent
    far=env.sim.model.vis.map.zfar*env.sim.model.stat.extent
    meters=near/(1.-np.asarray(buffer)*(1.-near/far))
    return rgb[::-1].copy(),meters[::-1].astype(np.float32)


class KitchenRecorder(EpisodeRecorder):
    def __init__(self,env,snapshot,success,env_config,output_dir,task_metadata=None,depth=False,hide_sites=True):
        add_wrist_camera(env)
        super().__init__(env,snapshot,success,env_config,task_metadata)
        configure_rendering(env,hide_sites)
        self.output_dir=Path(output_dir);self.output_dir.mkdir(parents=True,exist_ok=True)
        self.sensor_path=self.output_dir/'sensors.hdf5';self.depth=depth
        self.min_free_bytes=int(os.environ.get('KITCHEN_MIN_FREE_BYTES','0'))
        self.sensors=h5py.File(self.sensor_path,'w');self.sensor_rows=0
        self.sensors.attrs['site_groups_hidden']=bool(hide_sites)
        self.sensors.attrs['physical_grasp_definition']='bilateral opposing left_finger/right_finger collision groups; excludes palm/wrist; pad_only_grasp also stored'
        self.sensors.attrs['alignment']='observation BEFORE action[action_index]; both cameras same unchanged simulator state'
        self.sensors.attrs['control_frequency']=float(env.control_freq)
        self.sensors.create_dataset('action_index',shape=(0,),maxshape=(None,),dtype='i8')
        self.sensors.create_dataset('timestamp',shape=(0,),maxshape=(None,),dtype='f8')
        self.sensors.create_dataset('state',shape=(0,18),maxshape=(None,18),dtype='f4')
        self.sensors.create_dataset('mission_audit',shape=(0,),maxshape=(None,),dtype=h5py.string_dtype())
        self.sensors.attrs['mission_initial']=json.dumps(mission_snapshot(env))
        gripper=env.robots[0].gripper['right']
        declared=getattr(gripper,'measurement_drive_joint','finger_joint')
        self.sensors.attrs['measured_gripper_joint']=declared if declared.startswith(gripper.naming_prefix) else gripper.naming_prefix+declared
        self.sensors.attrs['measured_gripper_definition']='normalized actual declared drive joint angle within model joint range; not jaw width or a commanded operation'
        self.sensors.attrs['action_layout']=json.dumps(env.robots[0].composite_controller.get_action_info_dict())
        for key,spec in CAMERAS.items():
            g=self.sensors.create_group(key);g.attrs['camera_metadata']=json.dumps(camera_metadata(env,key))
            w,h=spec['width'],spec['height']
            g.attrs['rgb_encoding']='JPEG quality85; decode to RGB uint8 HWC'
            g.attrs['decoded_shape']=[h,w,3]
            g.create_dataset('rgb_jpeg',shape=(0,),maxshape=(None,),dtype=h5py.vlen_dtype(np.dtype('u1')))
            g.create_dataset('camera_to_world',shape=(0,4,4),maxshape=(None,4,4),dtype='f8')
            if key=='kinect' and depth:g.create_dataset('depth_m',shape=(0,h,w),maxshape=(None,h,w),chunks=(1,h,w),dtype='f2',compression='gzip',compression_opts=1)
    def _append(self,dataset,value):
        dataset.resize(self.sensor_rows+1,axis=0);dataset[self.sensor_rows]=value
    def step(self,action,phase=''):
        # Check during capture as well as before launching; reserve closeout space.
        if self.min_free_bytes and self.sensor_rows%20==0 and shutil.disk_usage(self.output_dir).free<self.min_free_bytes+32*1024**2:
            raise RuntimeError('Disk reserve reached during capture; stopping before further RGB writes')
        if self.sensor_rows!=len(self.actions):raise RuntimeError('Sensor/action alignment broken')
        for key,spec in CAMERAS.items():
            output=render_camera(self.env,key,depth=self.depth and key=='kinect')
            g=self.sensors[key]
            if self.depth and key=='kinect':
                rgb,depth=output;self._append(g['depth_m'],np.minimum(depth,65504).astype(np.float16))
            else:rgb=output
            buffer=io.BytesIO();Image.fromarray(rgb).save(buffer,format='JPEG',quality=85)
            self._append(g['rgb_jpeg'],np.frombuffer(buffer.getvalue(),dtype=np.uint8))
            if self.sensor_rows==0:Image.fromarray(rgb).save(self.output_dir/(key+'_first_frame.png'))
            cid=self.env.sim.model.camera_name2id(spec['name']);extrinsic=np.eye(4)
            extrinsic[:3,:3]=self.env.sim.data.cam_xmat[cid].reshape(3,3)
            extrinsic[:3,3]=self.env.sim.data.cam_xpos[cid]
            self._append(g['camera_to_world'],extrinsic)
        self._append(self.sensors['state'],measured_state(self.env))
        self._append(self.sensors['mission_audit'],json.dumps(mission_snapshot(self.env)))
        self._append(self.sensors['action_index'],len(self.actions))
        self._append(self.sensors['timestamp'],len(self.actions)/float(self.env.control_freq))
        self.sensor_rows+=1
        return super().step(action,phase)
    def save(self,path):
        self.sensors.attrs['mission_final']=json.dumps(mission_snapshot(self.env))
        self.sensors.flush();self.sensors.close()
        if self.sensor_rows!=len(self.actions):raise RuntimeError('Cannot save mismatched sensors/actions')
        path=super().save(path)
        if path.parent.resolve()!=self.sensor_path.parent.resolve():raise ValueError('Keep demo and sensors files together for portable links')
        with h5py.File(path,'a') as f:
            obs=f['data/demo_0'].create_group('obs')
            for key in CAMERAS:obs[key+'_image_jpeg']=h5py.ExternalLink(self.sensor_path.name,f'/{key}/rgb_jpeg')
            if self.depth:obs['kinect_depth_m']=h5py.ExternalLink(self.sensor_path.name,'/kinect/depth_m')
            obs['state']=h5py.ExternalLink(self.sensor_path.name,'/state')
            obs['timestamp']=h5py.ExternalLink(self.sensor_path.name,'/timestamp')
            obs['action_index']=h5py.ExternalLink(self.sensor_path.name,'/action_index')
        return path


def audit_sensor_alignment(path):
    path=Path(path);problems=[];streams={}
    with h5py.File(path) as f,h5py.File(path.with_name('sensors.hdf5')) as sensor:
        n=len(f['data/demo_0/actions']);freq=float(sensor.attrs['control_frequency'])
        if not np.array_equal(sensor['action_index'][:],np.arange(n)):problems.append('action indices mismatch')
        if len(sensor['timestamp'])!=n or not np.allclose(sensor['timestamp'][:],np.arange(n)/freq):problems.append('timestamp mismatch')
        for key,spec in CAMERAS.items():
            g=sensor[key];rgb=g['rgb_jpeg']
            indices=np.unique(np.linspace(0,n-1,min(n,12),dtype=int))
            samples=np.stack([np.asarray(Image.open(io.BytesIO(rgb[int(i)].tobytes())).convert('RGB')) for i in indices])
            if len(rgb)!=n or samples.shape[1:]!=(spec['height'],spec['width'],3):problems.append(key+' shape mismatch')
            if not np.isfinite(g['camera_to_world'][:]).all():problems.append(key+' nonfinite camera pose')
            variances=np.var(samples.reshape(len(samples),-1),axis=1)
            if np.min(variances)<1:problems.append(key+' blank image')
            streams[key]=dict(shape=[n,spec['height'],spec['width'],3],encoding='jpeg_quality85',minimum_sample_pixel_variance=float(variances.min()),metadata=json.loads(g.attrs['camera_metadata']))
        if 'depth_m' in sensor['kinect']:
            for frame in sensor['kinect/depth_m']:
                if not np.isfinite(frame).all() or not (frame>0).all():problems.append('invalid Kinect depth');break
    return dict(passed=not problems,issues=problems,streams=streams,frames=n,
                sensor_bytes=path.with_name('sensors.hdf5').stat().st_size)


def audit_camera_replay(path,make_env,success=None,sample_count=8,state_tolerance=1e-5):
    """Fresh initial-only action replay, comparing synchronized sampled camera frames."""
    path=Path(path);env=None
    report=dict(passed=False,initial_state_restored_once=True,subsequent_state_forcing=False)
    try:
        with h5py.File(path) as f:
            g=f['data/demo_0'];config=json.loads(f['data'].attrs['env_info'])
            states=g['states'][:];actions=g['actions'][:];final=g['final_state'][:]
            xml=g.attrs['model_file'];meta=json.loads(g.attrs['ep_meta'])
        env=make_env(config)
        if hasattr(env,'set_ep_meta'):env.set_ep_meta(meta)
        # Native kitchen environments may defer model construction until reset.
        if getattr(env,'model',None) is None or getattr(env,'sim',None) is None:env.reset()
        env.reset_from_xml_string(env.edit_model_xml(xml) if hasattr(env,'edit_model_xml') else xml)
        env.sim.reset();env.sim.set_state_from_flattened(states[0]);env.sim.forward()
        if hasattr(env,'initialize_sequence_history'):env.initialize_sequence_history()
        indices=set(np.linspace(0,len(actions)-1,min(sample_count,len(actions)),dtype=int).tolist())
        maximum=0.;comparisons=[];successes=[]
        with h5py.File(path.with_name('sensors.hdf5')) as sensors:
            configure_rendering(env,bool(sensors.attrs.get('site_groups_hidden',False)))
            for i,action in enumerate(actions):
                error=float(np.max(np.abs(env.sim.get_state().flatten()-states[i])))
                maximum=max(maximum,error)
                if i in indices:
                    for key in CAMERAS:
                        rgb=render_camera(env,key)
                        buffer=io.BytesIO();Image.fromarray(rgb).save(buffer,format='JPEG',quality=85)
                        fresh=np.asarray(Image.open(io.BytesIO(buffer.getvalue())).convert('RGB'),dtype=np.int16)
                        saved=np.asarray(Image.open(io.BytesIO(sensors[key+'/rgb_jpeg'][i].tobytes())).convert('RGB'),dtype=np.int16)
                        mae=float(np.abs(fresh-saved).mean())
                        comparisons.append(dict(action_index=i,camera=key,decoded_jpeg_mean_absolute_error=mae))
                env.step(action.copy())
                if success is not None:successes.append(bool(success(env)))
            maximum=max(maximum,float(np.max(np.abs(env.sim.get_state().flatten()-final))))
        report.update(maximum_state_error=maximum,comparisons=comparisons,replayed_actions=len(actions),
                      final_ten_step_success=all(successes[-10:]) if len(successes)>=10 else None)
        report['passed']=bool(np.isfinite(maximum) and maximum<=state_tolerance and comparisons and
                              all(c['decoded_jpeg_mean_absolute_error']<=1 for c in comparisons) and
                              (success is None or report['final_ten_step_success']))
    except Exception as exc:report['error']=f'{type(exc).__name__}: {exc}'
    finally:
        if env is not None:env.close()
    path.with_name('camera_replay_audit.json').write_text(json.dumps(report,indent=2))
    return report


def measured_state(env):
    """q7,dq7,normalized measured drive-joint angle, physical body twist3."""
    sim=env.sim
    names=[f'robot0_Actuator{i}' for i in range(1,8)]
    q=[float(sim.data.get_joint_qpos(name)) for name in names]
    dq=[float(sim.data.get_joint_qvel(name)) for name in names]
    gripper=env.robots[0].gripper['right']
    declared=getattr(gripper,'measurement_drive_joint','finger_joint')
    finger=declared if declared.startswith(gripper.naming_prefix) else gripper.naming_prefix+declared
    jid=sim.model.joint_name2id(finger);lo,hi=sim.model.jnt_range[jid]
    closedness=float(np.clip((sim.data.get_joint_qpos(finger)-lo)/(hi-lo),0,1))
    root='mobilebase0_base'
    R=sim.data.get_body_xmat(root).reshape(3,3)
    linear=R.T@sim.data.get_body_xvelp(root);angular=R.T@sim.data.get_body_xvelr(root)
    result=np.r_[q,dq,closedness,linear[:2],angular[2]].astype(np.float32)
    if result.shape!=(18,) or not np.isfinite(result).all():raise ValueError('Invalid measured proprioception')
    return result


def mission_snapshot(env):
    """Read-only privileged evidence; excluded from policy observation fields."""
    result=dict(base_xyz=env.sim.data.get_body_xpos('mobilebase0_base').tolist(),objects={},fixtures={},history={})
    if isinstance(getattr(env,'objects',None),dict):
        for name,obj in env.objects.items():
            body=getattr(env,'obj_body_id',{}).get(name)
            if body is not None:
                result['objects'][name]=dict(xyz=env.sim.data.body_xpos[body].tolist(),
                    physical_grasp=bilateral_finger_grasp(env,obj),
                    pad_only_grasp=bool(env._check_grasp(env.robots[0].gripper['right'],obj)))
    for key,value in getattr(env,'fixture_refs',{}).items():
        fixture=value[0] if isinstance(value,tuple) else value
        record=dict(name=fixture.name,xyz=np.asarray(fixture.pos).tolist(),type=type(fixture).__name__)
        joints=getattr(fixture,'door_joint_names',[])
        if joints:record['door_fraction']=fixture.get_joint_state(env,joints)
        result['fixtures'][key]=record
    for name in ('_sequence_opened','_sequence_transferred','_sequence_max_motion','_sequence_initial_distance','_sequence_min_distance','_sequence_initial_fixture_distance','_sequence_min_fixture_distance','_sequence_initial_closed'):
        if hasattr(env,name):
            value=getattr(env,name)
            result['history'][name]=value.item() if isinstance(value,np.generic) else value
    return result


def bilateral_finger_grasp(env,obj):
    gripper=env.robots[0].gripper['right']
    return bool(env._check_grasp([gripper.important_geoms['left_finger'],gripper.important_geoms['right_finger']],obj))


def analyze_camera_projection(path,object_name='obj'):
    """Geometry visibility diagnostic; does not prove absence of occlusion."""
    path=Path(path);report=dict(object_name=object_name,cameras={},scope='Projected object center only; inspect RGB for occlusion and full object/hand visibility')
    with h5py.File(path) as f,h5py.File(path.with_name('sensors.hdf5')) as sensors:
        phases=f['data/demo_0/phases'].asstr()[:]
        records=[json.loads(s) for s in sensors['mission_audit'].asstr()[:]]
        positions=np.asarray([m['objects'][object_name]['xyz'] for m in records])
        for key in CAMERAS:
            metadata=json.loads(sensors[key].attrs['camera_metadata']);K=np.asarray(metadata['intrinsics'])
            transforms=sensors[key+'/camera_to_world'][:]
            relative=positions-transforms[:,:3,3]
            coords=np.einsum('nji,nj->ni',transforms[:,:3,:3],relative)
            distance=-coords[:,2]
            pixels=np.c_[K[0,0]*coords[:,0]/np.maximum(distance,1e-9)+K[0,2],K[1,2]-K[1,1]*coords[:,1]/np.maximum(distance,1e-9)]
            left=80 if key=='wrist' else 0;right=560 if key=='wrist' else 640
            visible=(distance>0)&(pixels[:,0]>left)&(pixels[:,0]<right)&(pixels[:,1]>0)&(pixels[:,1]<360)
            by_phase={}
            for phase in sorted(set(phases)):
                mask=phases==phase
                by_phase[phase]=dict(frames=int(mask.sum()),projected_center_in_frame_fraction=float(visible[mask].mean()))
            report['cameras'][key]=dict(overall_projected_center_in_frame_fraction=float(visible.mean()),phases=by_phase)
    path.with_name('camera_projection_diagnostic.json').write_text(json.dumps(report,indent=2))
    return report
