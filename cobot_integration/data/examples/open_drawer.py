"""Privileged smoke teacher reusing the existing LabApt fixture expert.

It uses native drawer/handle geometry and the existing experimental motion
planner. Other kitchens may be refused by that planner; only successful,
independently replayed episodes qualify. Layout 100 remains transfer TEST data.
"""
import types
import numpy as np
from scipy.spatial.transform import Rotation


def run(ctx):
    env=ctx.env
    if type(env).__name__!='OpenDrawer':
        raise ValueError('Example teacher supports native OpenDrawer only')
    from cobot_integration.labapt.experts import LabAptExpert
    expert=LabAptExpert(env)
    def tick(self,pos,rot=None,grip=None,base=None):
        rot=self.rotation if rot is None else rot
        grip=self.grip if grip is None else grip;self.grip=grip
        ctrl=self.robot.part_controllers['right'];ctrl.update(force=True)
        delta_position=ctrl.origin_ori.T@(np.asarray(pos)-ctrl.ref_pos)
        world_rotation=Rotation.from_matrix(np.asarray(rot)@ctrl.ref_ori_mat.T).as_rotvec()
        delta_rotation=ctrl.origin_ori.T@world_rotation
        command=np.r_[delta_position,delta_rotation]/ctrl.output_max
        command=np.clip(command,-1,1)
        a=self.robot.create_action_vector({'right':command,'right_gripper':[grip],
            'base':np.zeros(3) if base is None else base,'base_mode':-1})
        result=ctx.step(a,self.phase);self.steps+=1
        return result
    expert.tick=types.MethodType(tick,expert)
    fixture=env.drawer
    handles=[g.get('name') for g in fixture.get_obj().iter('geom') if 'handle' in g.get('name','') and g.get('group','0')=='0']
    plans=[]
    for body in fixture.get_obj().iter('body'):
        joint=body.find('joint')
        hs=[g.get('name') for g in body.iter('geom') if g.get('name') in handles]
        if joint is not None and hs:plans.append((joint.get('name'),hs[0]))
    if not plans:raise ValueError('Drawer handle/joint not found')
    for joint,handle in plans:expert.open_fixture(fixture,joint,handle)
    position,orientation=expert.pose()
    for _ in range(30):expert.tick(position,orientation,-1)
