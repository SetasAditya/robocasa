"""Opt-in CoBot gripper using RoboLab's actual linked MuJoCo 2F85 asset.

Preserves all linkage equalities, contacts, actuator gains and force limits. This
is robot mechanism geometry, never an artificial object attachment. Existing
CobotKinova remains unchanged so old prototype diagnostics can still replay.
"""
from pathlib import Path
import numpy as np
from robosuite.models.grippers.gripper_model import GripperModel
from robosuite.models.grippers import GRIPPER_MAPPING
from robosuite.robots import register_robot_class
from cobot import CobotKinova
DEST=Path(__file__).parent/'assets/cobot_fourbar85.xml'

class CobotFourBar85(GripperModel):
    def __init__(self,idn=0):super().__init__(str(DEST),idn=idn)
    @property
    def dof(self):return 1
    @property
    def measurement_drive_joint(self):return self.correct_naming('right_driver_joint')
    @property
    def init_qpos(self):return np.zeros(8)
    def format_action(self,action):
        self.current_action=np.clip(self.current_action+.2*np.sign(np.asarray(action)),-1.,1.)
        return self.current_action.copy()
    @property
    def _important_geoms(self):
        return {'left_fingerpad':['left_pad1','left_pad2'],'right_fingerpad':['right_pad1','right_pad2'],
                'left_finger':['left_pad1','left_pad2','left_follower_collision_1'],
                'right_finger':['right_pad1','right_pad2','right_follower_collision_1']}

GRIPPER_MAPPING['CobotFourBar85']=CobotFourBar85

@register_robot_class('WheeledRobot')
class CobotKinovaFourBar(CobotKinova):
    @property
    def default_gripper(self):return {'right':'CobotFourBar85'}
