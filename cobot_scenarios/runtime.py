"""Simulator adapter. Never imported by registry / ledger commands."""
import hashlib, os, sys
from pathlib import Path
from .core import read, write_new, digest

def bootstrap(integration):
    integration=Path(integration).resolve()
    sys.path.insert(0,str(integration))
    if str(integration.parent) not in sys.path:sys.path.append(str(integration.parent))
    os.environ.setdefault('MUJOCO_GL','egl')
    import cobot_fourbar, kitchen_tasks
    register()

def register():
    import numpy as np
    import kitchen_tasks as kt
    from robocasa.environments.kitchen.kitchen import Kitchen, FixtureType, OU
    from robocasa.environments.kitchen.atomic.kitchen_doors import OpenCabinet,CloseCabinet
    from robocasa.environments.kitchen.atomic.kitchen_drawer import OpenDrawer,CloseDrawer
    from robocasa.environments.kitchen.atomic.kitchen_pick_place import PickPlaceCounterToSink,PickPlaceCounterToCabinet,PickPlaceCabinetToCounter,PickPlaceCounterToMicrowave,PickPlaceMicrowaveToCounter
    global REGISTERED
    if globals().get('REGISTERED'):return
    class SuiteMixin:
        def __init__(self,scenario,asset_root,**kwargs):
            self.scenario=scenario;self.asset_root=Path(asset_root)
            self.opened=False;self.transferred=False;self.mobile_distance=0.;self.start_base=None
            self.composite=scenario['task'] in ('cabinet_transfer','microwave_transfer')
            super().__init__(**kwargs)
        def _get_obj_cfgs(self):
            cfgs=super()._get_obj_cfgs()
            # Explicit objects only: native random distractors would leak splits.
            cfgs=[c for c in cfgs if c['name'] in ('obj','obj_second')]
            asset=self.scenario.get('object_asset')
            for c in cfgs:
                if asset:c['obj_groups']=str(self.asset_root/asset['path'])
                p=c['placement'];p['pos']=(self.scenario['variation']['region_x'],-1.)
                p['rotation']=np.deg2rad(self.scenario['variation']['object_yaw_deg'])
            if cfgs:
                source=cfgs[0]['placement']['fixture']
                # Background clutter belongs on the work surface, not packed
                # into the appliance/cabinet containing the retrieval target.
                if self.composite:source=self.counter
                for i,path in enumerate(self.scenario.get('distractor_assets',[])):
                    cfgs.append(dict(name=f'distractor_{i}',obj_groups=str(self.asset_root/path),graspable=True,
                        placement=dict(fixture=source,size=(.7,.3),pos=(None,-1.))))
            return cfgs
        def _create_objects(self):
            super()._create_objects()
            self.placement_adjustments=[]
            if self._ep_meta.get('object_cfgs'):return  # Frozen placement metadata wins.
            if self.scenario['env']!='SuiteMicrowaveUnload':return
            from .geometry import fitting_yaw
            cfg=next(c for c in self.object_cfgs if c['name']=='obj')
            obj=self.objects['obj'];placement=cfg['placement']
            region=self.microwave.sample_reset_region(env=self,min_size=obj.size)
            cfg['reset_region']=region
            usable=np.asarray(region['size'])[:2]-placement.get('margin',.04)
            requested=float(placement['rotation'])
            realized=fitting_yaw(obj.size,usable,requested)
            placement['rotation']=realized
            self.placement_adjustments.append(dict(object='obj',requested_yaw_deg=float(np.rad2deg(requested)),realized_yaw_deg=float(np.rad2deg(realized)),usable_region_m=usable.tolist(),object_size_m=np.asarray(obj.size).tolist(),reason='fit conservative object bounds inside appliance'))
        def _setup_scene(self):
            super()._setup_scene()
            if self.composite:getattr(self,'cab' if self.scenario['task']=='cabinet_transfer' else 'microwave').close_door(env=self)
            elif self.scenario['task'].endswith('_articulation'):
                fixture=self.fxtr if self.scenario['task']=='cabinet_articulation' else self.drawer
                fraction=self.scenario['variation']['initial_open_fraction']
                fixture.set_joint_state(fraction,fraction,self,fixture.door_joint_names)
        def initialize_sequence_history(self):
            self.opened=False;self.transferred=False;self.mobile_distance=0.
            self.start_base=self.sim.data.get_body_xpos('mobilebase0_base')[:2].copy()
        def _reset_internal(self):
            super()._reset_internal();self.initialize_sequence_history()
        def _post_action(self,action):
            if self.start_base is not None:
                self.mobile_distance=max(self.mobile_distance,float(np.linalg.norm(self.sim.data.get_body_xpos('mobilebase0_base')[:2]-self.start_base)))
            if self.composite:
                fixture=getattr(self,'cab' if self.scenario['task']=='cabinet_transfer' else 'microwave')
                self.opened=self.opened or fixture.is_open(env=self)
                self.transferred=self.transferred or (self.opened and super()._check_success())
            return super()._post_action(action)
        def _check_success(self):
            if self.mobile_distance<.3:return False
            native=bool(super()._check_success())
            if self.composite:
                fixture=getattr(self,'cab' if self.scenario['task']=='cabinet_transfer' else 'microwave')
                return bool(native and self.opened and self.transferred and fixture.is_closed(env=self))
            return native
        def get_ep_meta(self):
            meta=super().get_ep_meta();meta['scenario_id']=self.scenario['scenario_id']
            if self.composite:meta['lang']='Approach, open the fixture, '+meta['lang'].rstrip('.')+', then close it.'
            return meta
    class SurfaceRefs:
        EXCLUDE_LAYOUTS=[]
        def _setup_kitchen_references(self):
            Kitchen._setup_kitchen_references(self)
            counters=sorted([f for f in self.fixtures.values() if 'Counter' in type(f).__name__],key=lambda f:f.name)
            minimum=(.381,.381) if self.scenario['task']=='collect_two_stations' else (.2,.2)
            counters=[f for f in counters if f.get_reset_regions(self,top_size=minimum)]
            pairs=[(a,b) for a in counters for b in counters if a is not b and np.linalg.norm(np.asarray(a.pos)[:2]-np.asarray(b.pos)[:2])>.8]
            if not pairs:raise ValueError('No separated counter pair')
            a,b=pairs[self.scenario['seed']%len(pairs)]
            self.source=self.register_fixture_ref('source',dict(id=a.name));self.target=self.register_fixture_ref('target',dict(id=b.name));self.init_robot_base_ref=self.source
        def _surface_labels(self):
            anchors=[f for f in self.fixtures.values() if any(k in type(f).__name__ for k in ('Sink','Stove','Microwave','Fridge'))]
            def label(f):
                if 'island' in f.name:return 'island counter'
                if 'dining' in f.name:return 'dining table'
                if anchors:
                    nearest=min(anchors,key=lambda a:np.linalg.norm(np.asarray(a.pos)[:2]-np.asarray(f.pos)[:2]))
                    return 'counter next to the '+nearest.nat_lang
                return 'counter'
            source,target=label(self.source),label(self.target)
            if source==target:
                source='counter holding the target object'
                target='other counter at least 0.8 metres away'
            return source,target
        def get_ep_meta(self):
            meta=super().get_ep_meta();source_label,target_label=self._surface_labels()
            if self.scenario['task']=='collect_two_stations':
                meta['lang']=f'Collect the {self.get_obj_lang("obj")} on the {source_label} and the {self.get_obj_lang("obj_second")} on the {target_label} into the blue bin.'
            else:
                target=self.get_obj_lang('obj')
                if self.scenario.get('same_category_distractor'):
                    axis=np.array([np.cos(self.source.rot),np.sin(self.source.rot)])
                    delta=np.asarray(self.object_placements['obj'][0])[:2]-np.asarray(self.object_placements['distractor_0'][0])[:2]
                    if abs(float(axis@delta))<.03:raise ValueError('Ambiguous left/right target separation')
                    target=('leftmost' if axis@delta<0 else 'rightmost')+' '+self.scenario['object_category']+' (when facing the source counter)'
                meta['lang']=f'Move the {target} from the {source_label} to the {target_label}.'
            return meta
    class MicrowaveDirect:
        """Dish transfer without native food-on-an-extra-plate task assumptions."""
        def _get_obj_cfgs(self):
            source=self.counter if self.scenario['env']=='SuiteMicrowaveLoad' else self.microwave
            return [dict(name='obj',obj_groups='mug',graspable=True,placement=dict(fixture=source,size=None,pos=(0,-1.)))]
        def get_ep_meta(self):
            meta=Kitchen.get_ep_meta(self)
            target='inside the microwave' if self.scenario['env']=='SuiteMicrowaveLoad' else 'on the counter'
            meta['lang']=f'Place the {self.get_obj_lang("obj")} {target}.'
            return meta
        def _check_success(self):
            target=self.microwave if self.scenario['env']=='SuiteMicrowaveLoad' else self.counter
            inside=OU.obj_inside_of(self,'obj',target) if target is self.microwave else True
            return bool(inside and OU.check_obj_fixture_contact(self,'obj',target) and OU.gripper_obj_far(self,'obj'))
    bases={'SuiteSurface':(SuiteMixin,SurfaceRefs,kt.KitchenFetchSurface),'SuiteSelective':(SuiteMixin,SurfaceRefs,kt.KitchenFetchSurface),
      'SuiteCollectTwo':(SuiteMixin,SurfaceRefs,kt.KitchenCollectTwoStations),'SuiteSink':(SuiteMixin,PickPlaceCounterToSink),
      'SuiteOpenCabinet':(SuiteMixin,OpenCabinet),'SuiteCloseCabinet':(SuiteMixin,CloseCabinet),
      'SuiteOpenDrawer':(SuiteMixin,OpenDrawer),'SuiteCloseDrawer':(SuiteMixin,CloseDrawer),
      'SuiteCabinetStore':(SuiteMixin,PickPlaceCounterToCabinet),'SuiteCabinetRetrieve':(SuiteMixin,PickPlaceCabinetToCounter),
      'SuiteMicrowaveLoad':(SuiteMixin,MicrowaveDirect,PickPlaceCounterToMicrowave),'SuiteMicrowaveUnload':(SuiteMixin,MicrowaveDirect,PickPlaceMicrowaveToCounter)}
    for name,parents in bases.items():globals()[name]=type(name,parents,{'__module__':__name__})
    REGISTERED=True

def make_config(m,asset_root,render=True):
    import kitchen_tasks
    cfg=kitchen_tasks.environment_config(m['env'],m['seed'],m['layout'],m['style'],render=render)
    cfg.update(scenario=m,asset_root=str(Path(asset_root).resolve()))
    return cfg

def make_env(config):
    import robosuite
    return robosuite.make(**config)

def collisions(env):
    """Report penetrating robot/environment contacts, excluding robot self contacts and floor."""
    bad=[]
    for c in env.sim.data.contact[:env.sim.data.ncon]:
        if c.dist>=-.003:continue
        names=[env.sim.model.geom_id2name(int(g)) or '' for g in (c.geom1,c.geom2)]
        robot=[n.startswith(('robot0_','mobilebase0_','gripper0_')) for n in names]
        object_contact=any(any(n.startswith(obj+'_') for obj in env.objects) for n in names)
        if (sum(robot)==1 and not any('floor' in n.lower() for n in names)) or (object_contact and c.dist<-.005):
            bad.append(dict(geoms=names,penetration_m=float(-c.dist)))
    return bad

def apply_start(env,m):
    import numpy as np
    from .geometry import fixture_bounds
    v=m['variation'];base=env.sim.data.get_body_xpos('mobilebase0_base').copy()
    R=env.sim.data.get_body_xmat('mobilebase0_base').copy();rects=fixture_bounds(env)
    names=['mobilebase0_joint_mobile_forward','mobilebase0_joint_mobile_side']
    ids=[env.sim.model.joint_name2id(n) for n in names]
    axes=np.stack([env.sim.data.xaxis[j][:2] for j in ids],axis=1)
    original=np.array([float(env.sim.data.get_joint_qpos(n)) for n in names])
    yaw='mobilebase0_joint_mobile_yaw';initial_yaw=float(env.sim.data.get_joint_qpos(yaw))
    candidates=[]
    # Preserve legacy order, then cover the narrow free sectors it missed.
    angles=list(dict.fromkeys([0,15,-15,30,-30,45,-45,60,-60,90,-90,135,-135,180]+list(range(-180,180,5))))
    for degrees in angles:
        a=np.deg2rad(degrees);turn=np.array([[np.cos(a),-np.sin(a)],[np.sin(a),np.cos(a)]])
        target=base[:2]-turn@R[:2,0]*v['base_distance_m']
        if not any(x-.45<=target[0]<=xx+.45 and y-.45<=target[1]<=yy+.45 for x,y,xx,yy in rects):candidates.append(target)
    for attempt,target in enumerate(candidates,1):
        for n,value in zip(names,original+np.linalg.solve(axes,target-base[:2])):env.sim.data.set_joint_qpos(n,value)
        env.sim.data.set_joint_qpos(yaw,initial_yaw+np.deg2rad(v['base_heading_deg']))
        env.sim.forward()
        if not collisions(env):
            env.initialize_sequence_history()
            return dict(candidate_count=len(candidates),attempts=attempt,dock_xy=base[:2].tolist(),start_xy=target.tolist(),requested_distance_m=v['base_distance_m'],realized_distance_m=float(np.linalg.norm(target-base[:2])),heading_deg=v['base_heading_deg'])
    # Never freeze a colliding fallback. Restore the native state and fail closed.
    for n,value in zip(names,original):env.sim.data.set_joint_qpos(n,value)
    env.sim.data.set_joint_qpos(yaw,initial_yaw);env.sim.forward()
    raise ValueError(f'No collision-free start among {len(candidates)} footprint-clear candidates at requested distance and heading')

def prepare(m,assets,out,render=True):
    import numpy as np
    from kitchen_recording import add_wrist_camera,render_camera
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    if (out/'resolved.json').exists():raise ValueError('Scenario already frozen; use restore')
    if digest({k:v for k,v in m.items() if k!='scenario_id'})[:20]!=m['scenario_id']:raise ValueError('Manifest integrity failure')
    if m.get('object_asset'):
        asset=m['object_asset'];path=Path(assets)/asset['path']
        if hashlib.sha256(path.read_bytes()).hexdigest()!=asset['sha256']:raise ValueError('Object asset hash mismatch')
    cfg=make_config(m,assets,render);env=make_env(cfg)
    try:
        env.reset()
        dock=env.sim.data.get_body_xpos('mobilebase0_base')[:2].copy()
        start_sampling=apply_start(env,m)
        placement_adjustments=getattr(env,'placement_adjustments',[])
        from .geometry import fixture_bounds,route
        rects=fixture_bounds(env);start=env.sim.data.get_body_xpos('mobilebase0_base')[:2]
        points=np.asarray([[r[0],r[1]] for r in rects]+[[r[2],r[3]] for r in rects]+[dock.tolist(),start.tolist()])
        bounds=(*np.min(points,axis=0)-.6,*np.max(points,axis=0)+.6)
        path=route(start,dock,rects,bounds,goal_snap=.6)
        navigation=dict(status='screen_passed' if path else 'needs_review',method='conservative inflated fixture AABB, radius .45m; staging goal within .6m of native dock',path=path,goal=dock.tolist())
        # Update source XML so recorder canonicalization retains lighting.
        for light in env.model.worldbody.iter('light'):
            diffuse=np.fromstring(light.get('diffuse','0.7 0.7 0.7'),sep=' ')*m['variation']['lighting_scale']
            light.set('diffuse',' '.join(map(str,np.clip(diffuse,0,1))))
        add_wrist_camera(env)
        xml=env.model.get_xml();meta=env.get_ep_meta();state=env.sim.get_state().flatten().copy()
        env.set_ep_meta(meta);env.reset_from_xml_string(xml);env.sim.reset();env.sim.set_state_from_flattened(state);env.sim.forward();env.initialize_sequence_history()
        contacts=collisions(env)
        if env._check_success():raise ValueError('Task already successful at initialization')
        geometry={}
        for name,obj in env.objects.items():
            try:half_size=np.asarray(obj.get_bounding_box_half_size()).tolist()
            except (AttributeError,NotImplementedError):half_size=None
            geometry[name]=dict(half_extents_m=half_size,position=env.sim.data.body_xpos[env.obj_body_id[name]].tolist(),grasp_fit='pending')
        resolved=dict(object_geometry=geometry,schema_version=1,scenario_id=m['scenario_id'],request=m,config=cfg,ep_meta=meta,instruction=meta['lang'],xml_sha256=hashlib.sha256(xml.encode()).hexdigest(),state_sha256=hashlib.sha256(state.tobytes()).hexdigest(),fixtures={k:(v[0] if isinstance(v,tuple) else v).name for k,v in env.fixture_refs.items()},qualification=dict(initialization='failed' if contacts else 'passed',contacts=contacts,reachability='pending',navigation=navigation,instruction_review='pending',manipulation='pending'),runtime=runtime_identity())
        resolved['sampling']=dict(base=start_sampling,object_adjustments=placement_adjustments)
        (out/'model.xml').write_text(xml);np.save(out/'state.npy',state);write_new(out/'resolved.json',resolved)
        if render:
            from PIL import Image
            for camera in ('kinect','wrist'):Image.fromarray(render_camera(env,camera)).save(out/(camera+'.png'))
        return resolved
    finally:env.close()

def runtime_identity():
    import robocasa,robosuite,mujoco,subprocess
    root=Path(robocasa.__file__).resolve().parent.parent
    rev=subprocess.run(['git','-C',str(root),'rev-parse','HEAD'],capture_output=True,text=True)
    import kitchen_tasks,cobot,cobot_fourbar,kitchen_recording
    files=[Path(__file__),Path(__file__).with_name('geometry.py')]+[Path(module.__file__) for module in (kitchen_tasks,cobot,cobot_fourbar,kitchen_recording)]
    return dict(robocasa_commit=rev.stdout.strip(),robosuite=robosuite.__version__,mujoco=mujoco.__version__,source_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in files})

def restore(directory,render=True):
    import numpy as np
    directory=Path(directory);r=read(directory/'resolved.json');xml=(directory/'model.xml').read_text();state=np.load(directory/'state.npy')
    if hashlib.sha256(xml.encode()).hexdigest()!=r['xml_sha256'] or hashlib.sha256(state.tobytes()).hexdigest()!=r['state_sha256']:raise ValueError('Frozen bundle integrity failure')
    if runtime_identity()!=r['runtime']:raise ValueError('Runtime version mismatch')
    cfg=dict(r['config']);cfg['has_offscreen_renderer']=render
    # Metadata must exist before rebuilding Python object/fixture inventory.
    env=make_env(cfg);env.set_ep_meta(r['ep_meta']);env.reset()
    env.reset_from_xml_string(xml);env.sim.reset();env.sim.set_state_from_flattened(state);env.sim.forward();env.initialize_sequence_history()
    if not np.array_equal(env.sim.get_state().flatten(),state):raise ValueError('State restore mismatch')
    return env,r

def snapshot(env):
    import numpy as np
    from kitchen_recording import bilateral_finger_grasp
    name='obj' if 'obj' in env.objects else None
    return dict(base_xy=env.sim.data.get_body_xpos('mobilebase0_base')[:2].copy(),object_xyz=env.sim.data.body_xpos[env.obj_body_id[name]].copy() if name else np.zeros(3),grasped=bilateral_finger_grasp(env,env.objects[name]) if name else False,object_name=name or '')
