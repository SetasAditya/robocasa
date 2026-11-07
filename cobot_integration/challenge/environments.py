"""Custom LabApt tasks; native fixtures and object samplers remain unchanged.

Scoring consumes simulator truth privately. Controllers receive only the public
sensor contract in runner.py. All scenes require qualification and a witness.
"""
from copy import deepcopy
import numpy as np
from robocasa.environments.kitchen.kitchen import Kitchen, FixtureType, OU
from robosuite.utils.mjcf_utils import new_body, new_geom, new_site
from .catalog import PROFILES, recipe
from .scoring import ScoreState
from .goals import rack_seated, bowl_seated, released_rest, tray_guides


class LabAptChallenge(Kitchen):
    PROFILE = None

    def __init__(self, difficulty='easy', initialization_mode='workflow', operator_geometry=None, **kwargs):
        self.challenge_recipe = recipe(self.PROFILE, difficulty, initialization_mode)
        self.qualified_geometry = deepcopy(operator_geometry or {})
        self.score_state = ScoreState(self.challenge_recipe['stable_seconds'])
        self._challenge_ready = False
        super().__init__(**kwargs)

    @property
    def kind(self):
        return self.challenge_recipe['kind']

    @property
    def params(self):
        return self.challenge_recipe['parameters']

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        if self.layout_id != 100:
            raise ValueError('Challenge v1 requires measured layout 100')
        counter_id = 'dining_black_table_labapt' if self.kind in ('stack','nest','tray','wipe','plate','rack','recovery') else 'appliance_counter_labapt'
        self.counter = self.register_fixture_ref('counter', dict(id=counter_id))
        kind = self.kind
        if kind in ('cabinet', 'stack', 'nest', 'door'):
            # A fixed, measured cabinet removes fixture choice as a confound.
            matches = [f for f in self.fixtures.values() if f.name.startswith('left_base')]
            if len(matches) != 1:
                raise ValueError('Measured left-base cabinet not uniquely available')
            self.work_fixture = self.register_fixture_ref('work_fixture', dict(id=matches[0].name))
        elif kind == 'drawer':
            self.work_fixture = self.register_fixture_ref('work_fixture', dict(id=FixtureType.TOP_DRAWER))
        elif kind in ('rack', 'plate'):
            self.work_fixture = self.register_fixture_ref('work_fixture', dict(id=FixtureType.DISHWASHER))
        else:
            self.work_fixture = self.counter
        self.init_robot_base_ref = self.work_fixture

    def _setup_scene(self):
        super()._setup_scene()
        self._challenge_ready = False
        self.score_state.reset()
        if self.kind in ('cabinet', 'stack', 'nest', 'door', 'drawer'):
            opening = self.params.get('door_open_fraction', self.params.get('drawer_open_fraction', 1.))
            self.work_fixture.open_door(self, min=opening, max=opening)
        elif self.kind in ('rack', 'plate'):
            self.work_fixture.open_door(self, min=1., max=1.)
            self.work_fixture.slide_rack(self, value=1.)

    def _cfg(self, name, group, fixture, *, pos=(0., -0.6), size=(1.0, .50), **kw):
        return dict(name=name, obj_groups=group, graspable=None if group in ('plate', 'tray', 'oven_tray') else True,
                    placement=dict(fixture=fixture, size=size, pos=pos), **kw)

    def _get_obj_cfgs(self):
        r = self.challenge_recipe
        source = self.work_fixture if self.kind in ('cabinet', 'drawer', 'door') else self.counter
        cfg = [self._cfg('obj', r['obj_group'], source,
                         max_size=(None, None, None))]
        if self.kind in ('stack', 'nest'):
            cfg.append(self._cfg('support', 'bowl', self.work_fixture, object_scale=1.0))
            cfg[0]['object_scale'] = .7 if self.kind == 'nest' else .85
            if self.kind == 'stack':
                cfg[-1]['placement'].update(rotation=(np.pi, np.pi), rotation_axis='x')
        if self.kind in ('cabinet', 'rack'):
            n = self.params.get('occlusion', 1 if r['difficulty']=='easy' else 2)
            for i in range(n):
                cfg.append(self._cfg(f'protected_{i}', 'glass_cup' if self.kind=='rack' else 'mug', self.work_fixture,
                                     pos=((i-(n-1)/2)*.35, -1.), size=(.20, .20)))
        if self.kind == 'tray':
            cfg[0]['placement'].update(pos=(-.75, -.5), size=(.60, .50), rotation=(0.,0.))
        return cfg

    def _load_model(self, *args, **kwargs):
        super()._load_model(*args, **kwargs)
        # Native placement retries recursively call this override. Add apparatus once.
        previous = self.model.worldbody.find("./body[@name='challenge_target']")
        if previous is not None:
            self.model.worldbody.remove(previous)
        for parent in self.model.worldbody.iter():
            for site in list(parent.findall("site[@name='challenge_rack_slot']")):
                parent.remove(site)
        # A visible target on a clear region; no measured fixture changes.
        regions = self.counter.get_reset_regions(self, full_depth_region=True)
        eligible = [r for r in regions.values() if min(r['size'][:2]) > .15]
        if not eligible:
            raise ValueError('No clear counter region for challenge target')
        region = max(eligible, key=lambda r: r['size'][0]*r['size'][1])
        theta = float(self.counter.rot)
        self._counter_R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        self.target_center = np.asarray(self.counter.pos, dtype=float).copy()
        self.target_center[:2] += self._counter_R @ np.asarray(region['offset'])[:2]
        self.target_center[2] += region['offset'][2]
        body = new_body(name='challenge_target', pos=self.target_center,
                        quat=[np.cos(theta/2), 0, 0, np.sin(theta/2)])
        width = self.params.get('patch_width_m', .20)
        body.append(new_site(name='challenge_goal', pos=[0, 0, .002], size=[min(width/2, region['size'][0]/2-.005), min(.10, region['size'][1]/2-.005), .001],
                             type='box', rgba=[.2, .65, .9, .25]))
        if self.kind == 'tray':
            # Removable tabletop apparatus, explicitly experimental until fit is checked.
            obj = self.objects['obj']
            parts = tray_guides(obj.size, self.params['clearance_m'])
            for i, (pos, size) in enumerate(parts):
                body.append(new_geom(name=f'challenge_guide_{i}', type='box', pos=pos, size=size,
                                     rgba=[.1, .3, .7, 1], group=1))
        self.model.worldbody.append(body)
        if self.kind in ('rack','plate') and self.qualified_geometry.get('slot'):
            slot = self.qualified_geometry['slot']
            joint_name = self.work_fixture._joint_names['rack']
            bodies = [b for b in self.model.worldbody.iter('body') if any(j.get('name')==joint_name for j in b.findall('joint'))]
            if len(bodies)!=1:
                raise ValueError('Cannot attach slot marker to moving rack')
            bodies[0].append(new_site(name='challenge_rack_slot', pos=slot['center'],
                size=slot['half_extents'], type='box', rgba=[.2,.65,.9,.25]))

    def _reset_internal(self):
        self._challenge_ready = False
        super()._reset_internal()
        self.sim.forward()
        self.protected_initial = {n: self._pose(n) for n in self.objects if n.startswith('protected_')}
        self.initial_door = self._opening() if self.kind == 'door' else 0.
        self.initial_target = self._pose('obj')[0]
        self._challenge_ready = True
        self.score_state.reset()

    def _pose(self, name):
        bid = self.obj_body_id[name]
        return (self.sim.data.body_xpos[bid].copy(), self.sim.data.body_xmat[bid].reshape(3, 3).copy())

    def _opening(self):
        values = self.work_fixture.get_joint_state(self, self.work_fixture.door_joint_names).values()
        return min(values)

    def _quiet(self, name):
        body = self.objects[name].root_body
        return (np.linalg.norm(self.sim.data.get_body_xvelp(body)) < self.challenge_recipe['linear_speed_limit_m_s']
                and np.linalg.norm(self.sim.data.get_body_xvelr(body)) < self.challenge_recipe['angular_speed_limit_rad_s'])

    def _protected(self):
        r = self.challenge_recipe
        return all(np.linalg.norm(self._pose(n)[0]-p) <= r['protected_displacement_m']
                   and np.arccos(np.clip(np.dot(self._pose(n)[1][:, 2], R[:, 2]), -1, 1)) <= r['protected_tilt_rad']
                   for n, (p, R) in self.protected_initial.items())

    def _contacts(self):
        object_geoms = {self.sim.model.geom_name2id(g) for o in self.objects.values() for g in o.contact_geoms}
        return [c for c in self.sim.data.contact if c.geom1 in object_geoms or c.geom2 in object_geoms]

    def _rack_local(self, name):
        j = self.sim.model.joint_name2id(self.work_fixture._joint_names['rack'])
        b = self.sim.model.jnt_bodyid[j]
        p, _ = self._pose(name)
        R = self.sim.data.body_xmat[b].reshape(3, 3)
        return R.T @ (p-self.sim.data.body_xpos[b]), R

    def _facts(self):
        p, R = self._pose('obj')
        upright = R[2, 2] > np.cos(.20)
        released = OU.gripper_obj_far(self, 'obj', th=.12) and not self.check_contact(self.robots[0].gripper['right'], self.objects['obj'])
        kind = self.kind
        cells = []
        if kind in ('cabinet', 'drawer', 'door', 'recovery'):
            target_xy = self._counter_R.T @ (p[:2]-self.target_center[:2])
            in_target = kind != 'recovery' or max(abs(target_xy)) < .10
            goal = OU.check_obj_fixture_contact(self, 'obj', self.counter) and in_target and (upright or kind=='drawer')
            if kind == 'door':
                goal &= self.score_state.articulated
        elif kind in ('rack', 'plate'):
            local, rack_R = self._rack_local('obj')
            # Recorded native rack bounds/slot center must be supplied by qualification.
            geometry = getattr(self, 'qualified_geometry', {})
            slot = geometry.get('slot')
            goal = False
            if slot:
                delta = local-np.asarray(slot['center'])
                bounds = np.asarray(slot['half_extents'])
                axis = int(slot.get('plate_normal_axis', 0))
                alignment = abs(float((rack_R.T@R[:, 2])[axis])) if kind=='plate' else float(R[2, 2])
                goal = rack_seated(delta=delta, half_extents=bounds, orientation_alignment=alignment,
                                   rack_contact=self.work_fixture.check_rack_contact(self, 'obj'))
        elif kind in ('stack', 'nest'):
            q, Q = self._pose('support')
            dz = p[2]-q[2]
            geom = getattr(self, 'qualified_geometry', {})
            interval = geom.get('seated_height_interval_m')
            goal = bowl_seated(height=dz, height_interval=interval,
                xy_distance=np.linalg.norm(p[:2]-q[:2]), tolerance=self.params['alignment_tolerance_m'],
                target_upright=upright, support_upright=(Q[2, 2] < -np.cos(.20) if kind=='stack' else Q[2, 2] > np.cos(.20)),
                contact=self.check_contact(self.objects['obj'], self.objects['support']),
                both_inside=OU.obj_inside_of(self, 'support', self.work_fixture) and OU.obj_inside_of(self, 'obj', self.work_fixture),
                support_quiet=self._quiet('support'), support_released=OU.gripper_obj_far(self, 'support', th=.12))
        elif kind == 'tray':
            local = self._counter_R.T @ (p[:2]-self.target_center[:2])
            goal = bool(abs(local[0]) < self.params['clearance_m']/2 and abs(local[1]) < .025
                        and abs(float(np.dot(R[:2, 0], self._counter_R[:, 0]))) > np.cos(.10)
                        and upright and self.check_contact(self.objects['obj'], 'challenge_guide_2')
                        and OU.check_obj_fixture_contact(self, 'obj', self.counter))
        else:
            width = self.params['patch_width_m']
            gripped = self.check_contact(self.robots[0].gripper['right'], self.objects['obj'])
            # Contact points, rather than sponge origin or image overlap, define coverage.
            if gripped:
                gids = {self.sim.model.geom_name2id(g) for g in self.objects['obj'].contact_geoms}
                fx = {self.sim.model.geom_name2id(g) for g in self.counter.contact_geoms}
                for c in self.sim.data.contact:
                    if (c.geom1 in gids and c.geom2 in fx) or (c.geom2 in gids and c.geom1 in fx):
                        xy = self._counter_R.T @ (np.asarray(c.pos)[:2]-self.target_center[:2])
                        if abs(xy[0]) < width/2 and abs(xy[1]) < .1:
                            ix = min(9, int((xy[0]/width+.5)*10))
                            iy = min(9, int((xy[1]/.2+.5)*10))
                            cells.append(iy*10+ix)
            goal = len(self.score_state.covered | set(cells)) >= 90 and OU.check_obj_fixture_contact(self, 'obj', self.counter)
        body = self.objects['obj'].root_body
        rest = released_rest(in_destination=goal, released=released,
            linear_speed=np.linalg.norm(self.sim.data.get_body_xvelp(body)),
            angular_speed=np.linalg.norm(self.sim.data.get_body_xvelr(body)),
            linear_limit=self.challenge_recipe['linear_speed_limit_m_s'],
            angular_limit=self.challenge_recipe['angular_speed_limit_rad_s'])
        return dict(goal=rest, protected=self._protected(),
                    penetration=any(c.dist < -self.challenge_recipe['max_penetration_m'] for c in self._contacts()),
                    cells=cells, articulated=bool(kind=='door' and self._opening() > self.initial_door+.1))

    def _post_action(self, action):
        if self._challenge_ready:
            self.score_state.advance(float(self.sim.data.time), **self._facts())
        return super()._post_action(action)

    def _check_success(self):
        return bool(self._challenge_ready and self.score_state.success)

    def challenge_state(self):
        return dict(score=self.score_state.snapshot(),
                    protected={n: [p.tolist(), R.tolist()] for n, (p, R) in self.protected_initial.items()},
                    initial_door=self.initial_door, initial_target=self.initial_target.tolist(),
                    geometry=deepcopy(getattr(self, 'qualified_geometry', {})))

    def restore_challenge_state(self, state):
        self.score_state.restore(state['score'])
        self.protected_initial = {n: (np.asarray(p), np.asarray(R)) for n, (p, R) in state['protected'].items()}
        self.initial_door = state['initial_door']
        self.initial_target = np.asarray(state['initial_target'])
        self.qualified_geometry = deepcopy(state['geometry'])
        self._challenge_ready = True

    def get_ep_meta(self):
        meta = super().get_ep_meta()
        meta['lang'] = self.challenge_recipe['instruction']
        meta['challenge_recipe'] = deepcopy(self.challenge_recipe)
        return meta


# Explicit environment names; no existing RoboCasa class is replaced.
for _profile in PROFILES:
    globals()['LabApt'+_profile] = type('LabApt'+_profile, (LabAptChallenge,),
                                      {'PROFILE': _profile, '__module__': __name__})
