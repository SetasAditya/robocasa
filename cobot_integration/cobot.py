"""CoBot prototype: existing chassis with robosuite Kinova3 / Robotiq85.

Uses bounded planar velocity actuators. Not certified equivalent to Isaac dynamics.
"""
from pathlib import Path
import numpy as np
from robosuite.models.bases import register_base
from robosuite.models.bases.mobile_base_model import MobileBaseModel
from robosuite.models.robots.manipulators.kinova3_robot import Kinova3
from robosuite.robots import register_robot_class
from robosuite.controllers import load_composite_controller_config

@register_base
class CobotMobileBase(MobileBaseModel):
    def __init__(self,idn=0):
        super().__init__(str(Path(__file__).parent/'assets/cobot_base_camera_aperture_v2.xml'),idn=idn)
    @property
    def top_offset(self): return np.zeros(3)
    @property
    def horizontal_radius(self): return 0.42

@register_robot_class('WheeledRobot')
class CobotKinova(Kinova3):
    def __init__(self,idn=0):
        super().__init__(idn=idn)
        # Align the prototype's arm bounds with RoboLab's mobile arm profile.
        for i,bound in [(2,2.24),(4,2.57),(6,2.09)]:
            j=self.worldbody.find(f".//joint[@name='{self.naming_prefix}Actuator{i}']")
            j.set('range',f'-{bound} {bound}')
        for i,motor in enumerate(self.actuator,1):
            effort=39 if i<=4 else 9
            motor.set('ctrlrange',f'-{effort} {effort}')
    @property
    def default_base(self): return 'CobotMobileBase'
    @property
    def default_arms(self): return {'right':'Kinova3'}
    @property
    def init_qpos(self):
        return np.array([0.,0.26179939,np.pi,-2.26892803,0.,0.95993109,np.pi/2])
    @property
    def _horizontal_radius(self): return 0.42

def controller_config():
    cfg=load_composite_controller_config(robot='PandaOmron')
    cfg['body_parts'].pop('torso',None)
    return cfg

def frame_viewer(env):
    """Frame the physical CoBot after kitchen reset instead of the world origin."""
    renderer = getattr(env, 'viewer', None)
    if renderer is None or not hasattr(renderer, 'camera_config'):
        return
    center = env.sim.data.get_body_xpos('mobilebase0_base') + np.array([0.,0.,0.7])
    config = dict(lookat=center.tolist(), distance=2.8, azimuth=135., elevation=-25.)
    renderer.camera_config = config
    renderer.set_camera(-1)
    handle = renderer.viewer
    if handle is not None:
        with handle.lock():
            handle.cam.type = 0
            handle.cam.fixedcamid = -1
            for key,value in config.items():
                setattr(handle.cam,key,value)
