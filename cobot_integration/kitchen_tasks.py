"""CoBot configuration and physical surface/bin extensions used by the suite.

Fixture tasks, compositional sequence history and variation selection live in
cobot_scenarios.runtime. No legacy campaign presets or task-specific experts.
"""
import numpy as np
from robocasa.environments.kitchen.kitchen import Kitchen, FixtureType, OU
from robosuite.utils.mjcf_utils import new_body, new_geom

CAMERAS=('mobilebase0_kinect','robot0_eye_in_hand')

def environment_config(task,seed,layout=1,style=1,render=False,images=False,image_size=256,**overrides):
    """CoBot settings shared by registered Suite environments at 20 Hz.

    Paired collection injects its calibrated simulation wrist camera separately;
    images=True uses legacy native observations and is not that camera contract.
    """
    from cobot import controller_config
    import cobot_fourbar  # Robot and physical linked hand registration.
    cfg=controller_config()
    cfg['body_parts']['right']['input_type']='absolute'
    cfg['body_parts']['right']['input_ref_frame']='world'
    kw=dict(env_name=task,robots='CobotKinovaFourBar',controller_configs=cfg,
            seed=int(seed),layout_ids=[int(layout)],style_ids=[int(style)],
            obj_registries=['objaverse','lightwheel'],translucent_robot=False,
            initialization_noise=None,control_freq=20,ignore_done=True,
            has_renderer=False,has_offscreen_renderer=bool(render or images),
            use_camera_obs=bool(images),camera_names=list(CAMERAS),
            camera_heights=image_size,camera_widths=image_size,
            robot_spawn_deviation_pos_x=.15,robot_spawn_deviation_pos_y=.05,
            robot_spawn_deviation_rot=0.)
    kw.update(overrides)
    return kw

class KitchenFetchSurface(Kitchen):
    """Carry a named household object from dining table to a distant counter."""
    EXCLUDE_LAYOUTS=Kitchen.DINING_COUNTER_EXCLUDED_LAYOUTS

    def __init__(self,obj_groups='glass_cup',distractors=False,**kwargs):
        self.fetch_obj_groups=obj_groups
        self.fetch_distractors=distractors
        super().__init__(**kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.source=self.register_fixture_ref('source',dict(id=FixtureType.DINING_COUNTER))
        candidates=[f for f in self.fixtures.values() if 'Counter' in type(f).__name__ and f is not self.source and np.linalg.norm(np.asarray(f.pos)[:2]-np.asarray(self.source.pos)[:2])>.8]
        if not candidates:raise ValueError('No distinct counter at least .8m from dining counter')
        target=sorted(candidates,key=lambda f:np.linalg.norm(np.asarray(f.pos)[:2]-np.asarray(self.source.pos)[:2]))[0]
        self.target=self.register_fixture_ref('target',dict(id=target.name))
        self.init_robot_base_ref=self.source

    def _setup_scene(self):
        super()._setup_scene()
        # Native layout2 island placement can initialize the folded arm through
        # an object on its left. Choose a safe left-of-object base anchor before
        # native reset settling; recorded initial poses remain authoritative.
        if self.source.name.startswith('island') and 'init_robot_base_pos' not in self._ep_meta:
            anchor=np.asarray(self.init_robot_base_pos_anchor).copy()
            anchor[0]=float(self.object_placements['obj'][0][0])-.17
            theta=float(self.source.rot);outward=np.array([np.sin(theta),-np.cos(theta)])
            front=np.asarray(self.source.pos)[:2]+outward*(self.source.size[1]/2+.60)
            anchor[:2]+=outward*np.dot(front-anchor[:2],outward)
            self.init_robot_base_pos_anchor=anchor

    def _get_obj_cfgs(self):
        cfg=[dict(name='obj',obj_groups=self.fetch_obj_groups,graspable=True,init_robot_here=True,
                  placement=dict(fixture=self.source,sample_region_kwargs=dict(full_depth_region=True),size=(.3,.3),pos=(0,-1.)))]
        if self.fetch_distractors:
            cfg.append(dict(name='distractor',obj_groups='tupperware',graspable=True,
                            placement=dict(fixture=self.source,reuse_region_from='obj',size=(.25,.25),pos=(.5,-1.))))
        return cfg

    def get_ep_meta(self):
        meta=super().get_ep_meta()
        meta['lang']=f'Fetch the {self.get_obj_lang("obj")} from the dining table and put it on the kitchen counter.'
        meta['task_extension']='KitchenFetchSurface on native RoboCasa kitchen'
        meta['distractor_initial_xyz']={n:p.tolist() for n,p in getattr(self,'_distractor_initial',{}).items()}
        return meta

    def _reset_internal(self):
        super()._reset_internal()
        baseline=self._ep_meta.get('distractor_initial_xyz')
        self._distractor_initial=({n:np.asarray(p,dtype=float) for n,p in baseline.items()} if baseline is not None else {n:self.sim.data.body_xpos[self.obj_body_id[n]].copy() for n in self.objects if n.startswith('distractor')})

    def _check_success(self):
        untouched=all(np.linalg.norm(self.sim.data.body_xpos[self.obj_body_id[n]]-p)<.04 for n,p in getattr(self,'_distractor_initial',{}).items())
        return bool(untouched and OU.check_obj_fixture_contact(self,'obj',self.target) and OU.gripper_obj_far(self,'obj'))

class KitchenBinPlacement(KitchenFetchSurface):
    """A physical open box on a real native counter, not a renamed bowl."""
    def _load_model(self,*args,**kwargs):
        super()._load_model(*args,**kwargs)
        theta=float(self.target.rot)
        R=np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
        regions=self.target.get_reset_regions(self,full_depth_region=True)
        candidates=[r for r in regions.values() if r['size'][0]>.38 and r['size'][1]>.38]
        if not candidates:raise ValueError('No unobstructed native countertop region fits physical bin')
        source=np.asarray(self.object_placements['obj'][0])
        region=min(candidates,key=lambda r:np.linalg.norm(np.asarray(self.target.pos)[:2]+R@np.asarray(r['offset'])[:2]-source[:2]))
        center=np.array(self.target.pos,dtype=float)
        local=np.array(region['offset'],dtype=float)
        local[1]+=-region['size'][1]/2+.22
        center[:2]+=R@local[:2]
        center[2]+=local[2]
        self.bin_floor_xyz=center.copy()
        body=new_body(name='collection_bin',pos=center,quat=[np.cos(theta/2),0,0,np.sin(theta/2)])
        parts=[([0,0,.006],[.18,.18,.006]),([-.174,0,.09],[.006,.18,.09]),([.174,0,.09],[.006,.18,.09]),([0,-.174,.09],[.168,.006,.09]),([0,.174,.09],[.168,.006,.09])]
        for i,(p,size) in enumerate(parts):
            body.append(new_geom(name=f'collection_bin_part{i}',type='box',pos=p,size=size,rgba=[.12,.32,.55,1],group=1,friction=[1,.005,.0001]))
        self.model.worldbody.append(body)
    def _inside_bin(self,name):
        bid=self.obj_body_id[name];pos=self.sim.data.body_xpos[bid]
        theta=float(self.target.rot);R=np.array([[np.cos(theta),np.sin(theta)],[-np.sin(theta),np.cos(theta)]])
        xy=R@(pos[:2]-self.bin_floor_xyz[:2])
        return bool(np.max(np.abs(xy))<.15 and self.bin_floor_xyz[2]<pos[2]<self.bin_floor_xyz[2]+.22 and self.check_contact(self.objects[name],'collection_bin_part0') and OU.gripper_obj_far(self,name))
    def _check_success(self):return hasattr(self,'bin_floor_xyz') and self._inside_bin('obj')
    def get_ep_meta(self):
        meta=super().get_ep_meta();meta['lang']=f'Fetch the {self.get_obj_lang("obj")} and put it inside the blue bin on the kitchen counter.'
        meta['task_extension']='KitchenBinPlacement: physical five-piece open box on native counter'
        if hasattr(self,'bin_floor_xyz'):meta['bin_floor_xyz']=self.bin_floor_xyz.tolist()
        return meta

class KitchenCollectTwoStations(KitchenBinPlacement):
    def __init__(self,second_obj_groups='can',**kwargs):
        self.second_obj_groups=second_obj_groups
        super().__init__(**kwargs)
    def _get_obj_cfgs(self):
        cfg=super()._get_obj_cfgs()
        cfg.append(dict(name='obj_second',obj_groups=self.second_obj_groups,graspable=True,
                        placement=dict(fixture=self.target,size=(.2,.2),pos=(.7,-1.))))
        return cfg
    def _check_success(self):
        return hasattr(self,'bin_floor_xyz') and self._inside_bin('obj') and self._inside_bin('obj_second')
    def get_ep_meta(self):
        meta=super().get_ep_meta();meta['lang']=f'Collect the {self.get_obj_lang("obj")} from the dining table and the {self.get_obj_lang("obj_second")} from the kitchen counter into the blue bin.'
        meta['task_extension']='KitchenCollectTwoStations on native dining and kitchen counter'
        return meta
