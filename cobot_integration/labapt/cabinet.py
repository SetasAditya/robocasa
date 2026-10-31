"""Cabinet-specific moving-handle recovery; normal actions only."""
import numpy as np
from scipy.spatial.transform import Rotation
from .experts import LabAptExpert

class LabAptCabinetExpert(LabAptExpert):

    def open_fixture(self, fixture, joint, handle, regrasp=0):
        e = self.env
        jid = e.sim.model.joint_name2id(joint)
        gid = e.sim.model.geom_name2id(handle)
        h = e.sim.data.get_geom_xpos(handle).copy()
        Rf = e.sim.data.get_body_xmat(fixture.root_body).copy()
        out = Rf @ np.array([0.0, -1.0, 0.0])
        if int(e.sim.model.jnt_type[jid]) != 2:
            out = Rotation.from_rotvec(e.sim.data.xaxis[jid] * float(e.sim.data.get_joint_qpos(joint))).as_matrix() @ out
        goal = h[:2] + out[:2] * 0.68
        self.phase = 'base_clearance'
        current, Q = self.pose()
        raised = current.copy()
        raised[2] = max(current[2], 1.08)
        self.move(raised, -1, n=100, hold=25, rot=Q)
        self.navigate(goal)
        h = e.sim.data.get_geom_xpos(handle).copy()
        out = Rf @ np.array([0.0, -1.0, 0.0])
        if int(e.sim.model.jnt_type[jid]) != 2:
            out = Rotation.from_rotvec(e.sim.data.xaxis[jid] * float(e.sim.data.get_joint_qpos(joint))).as_matrix() @ out
        axis = e.sim.data.get_geom_xmat(handle)[:, 2] if int(e.sim.model.geom_type[gid]) == 5 else Rf[:, 2]
        if int(e.sim.model.geom_type[gid]) == 6:
            axis = e.sim.data.get_geom_xmat(handle)[:, int(np.argmax(e.sim.model.geom_size[gid]))]
        z = -out
        x = np.cross(axis, z)
        x = x / np.linalg.norm(x)
        if np.dot(x, Rf[:, 0]) < -0.1:
            x = -x
        R = np.column_stack([x, np.cross(z, x), z])
        self.rotation = R
        self.phase = 'handle_pregrasp'
        before_approach = float(e.sim.data.get_joint_qpos(joint))
        try:
            self.posture(h + out * 0.16, R)
        except RuntimeError as exc:
            moved = abs(float(e.sim.data.get_joint_qpos(joint)) - before_approach) > 0.05
            if regrasp >= 2 or not moved or 'physical tracking' not in str(exc):
                raise
            self.phase = 'moving_handle_recovery'
            p, Q = self.pose()
            for _ in range(40):
                self.tick(p, Q, -1)
            self.move(p + out * 0.12, -1, n=70, hold=25, rot=Q)
            if e._check_success():
                return True
            return self.open_fixture(fixture, joint, handle, regrasp + 1)
        self.phase = 'handle_engage'
        axis = e.sim.data.xaxis[jid].copy()
        slide = int(e.sim.model.jnt_type[jid]) == 2
        for distance in np.r_[np.linspace(0.16, 0.0, 110), np.zeros(45)]:
            angle = float(e.sim.data.get_joint_qpos(joint)) - before_approach
            turn = np.eye(3) if slide else Rotation.from_rotvec(axis * angle).as_matrix()
            goal = e.sim.data.get_geom_xpos(handle).copy() + turn @ out * distance
            self.tick(goal, turn @ R, -1)
        err = float(np.linalg.norm(self.pose()[0] - e.sim.data.get_geom_xpos(handle)))
        if err > 0.035:
            raise RuntimeError(f'LabApt handle tracking error {err:.3f}m')
        self.phase = 'handle_close'
        for _ in range(100):
            angle = float(e.sim.data.get_joint_qpos(joint)) - before_approach
            turn = np.eye(3) if slide else Rotation.from_rotvec(axis * angle).as_matrix()
            self.tick(e.sim.data.get_geom_xpos(handle).copy(), turn @ R, 1)
        h = e.sim.data.get_geom_xpos(handle).copy()
        out = turn @ out
        groups = [self.robot.gripper['right'].important_geoms[k] for k in ('left_finger', 'right_finger')]
        if not e._check_grasp(groups, [handle]):
            raise RuntimeError('LabApt handle lacks bilateral physical grasp')
        lo, hi = e.sim.model.jnt_range[jid]
        target = 0.98 * (lo if abs(lo) > abs(hi) else hi)
        q0 = float(e.sim.data.get_joint_qpos(joint))
        p0, Q = self.pose()
        anchor = e.sim.data.xanchor[jid].copy()
        axis = e.sim.data.xaxis[jid].copy()
        slide = int(e.sim.model.jnt_type[jid]) == 2
        self.phase = 'physical_open'
        b0 = self.base().copy()
        lost = 0
        checkpoint = q0
        initial_rotation = e.sim.data.get_body_xmat('mobilebase0_base').copy()
        initial_yaw = np.arctan2(initial_rotation[1, 0], initial_rotation[0, 0])
        if slide:
            target = -1.05 * float(fixture.size[1]) * 0.55
        for step in range(1800):
            actual = float(e.sim.data.get_joint_qpos(joint))
            if abs(actual - target) < (0.003 if slide else 0.045) or (not slide and abs(actual) >= 0.94 * max(abs(lo), abs(hi))):
                break
            lead = 0.008 if slide else 0.055
            q = actual + np.clip(target - actual, -lead, lead)
            turn = Rotation.from_rotvec(axis * (q - q0)).as_matrix()
            p = p0 + axis * (q - q0) if slide else anchor + turn @ (p0 - anchor)
            if slide:
                desired_base = b0[:2] + 0.8 * (p[:2] - p0[:2])
            else:
                handle_goal = anchor + turn @ (h - anchor)
                desired_base = (handle_goal + turn @ out * 0.68)[:2]
                inward = float(np.dot(desired_base - b0[:2], out[:2]))
                if inward < 0:
                    desired_base -= inward * out[:2]
            worldvel = np.clip((desired_base - self.base()[:2]) * 1.5, -0.08, 0.08)
            controller = self.robot.part_controllers['base']
            _, ori = controller.get_base_pose()
            theta = np.arctan2(ori[1, 0], ori[0, 0]) - np.arctan2(controller.init_ori[1, 0], controller.init_ori[0, 0])
            ids = [e.sim.model.joint_name2id('mobilebase0_joint_mobile_' + n) for n in ('forward', 'side')]
            axes = np.stack([e.sim.data.xaxis[j][:2] for j in ids], axis=1)
            v = np.linalg.solve(axes, worldvel)
            v = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]) @ v
            body_rotation = e.sim.data.get_body_xmat('mobilebase0_base')
            yaw = np.arctan2(body_rotation[1, 0], body_rotation[0, 0])
            direction = h[:2] - b0[:2] if slide else handle_goal[:2] - self.base()[:2]
            initial_direction = h[:2] - b0[:2]
            wanted_yaw = initial_yaw + np.arctan2(initial_direction[0] * direction[1] - initial_direction[1] * direction[0], np.dot(initial_direction, direction))
            yaw_error = np.arctan2(np.sin(wanted_yaw - yaw), np.cos(wanted_yaw - yaw))
            self.tick(p, Q if slide else turn @ Q, 1, base=np.r_[v, np.clip(yaw_error, -0.25, 0.25)])
            held = e._check_grasp(groups, [handle])
            lost = 0 if held else lost + 1
            if step % 100 == 0:
                print('LABAPT_JOINT', joint, actual, 'target', target, 'held', held, 'tcp_error', float(np.linalg.norm(self.pose()[0] - p)), flush=True)
            stalled = step > 0 and step % 250 == 0 and (abs(actual - checkpoint) < (0.005 if slide else 0.04))
            if step % 250 == 0:
                checkpoint = actual
            if lost > 40 or stalled:
                if regrasp >= 2:
                    raise RuntimeError('Handle progress failed after two physical regrasp attempts')
                p, Q = self.pose()
                for _ in range(50):
                    self.tick(p, Q, -1)
                now = float(e.sim.data.get_joint_qpos(joint))
                retreat = out if slide else Rotation.from_rotvec(axis * (now - q0)).as_matrix() @ out
                self.move(p + retreat * 0.12, -1, n=80, hold=30, rot=Q)
                current_joint = float(e.sim.data.get_joint_qpos(joint))
                if bool(e._check_success()) or (not slide and abs(current_joint) >= 0.94 * max(abs(lo), abs(hi))):
                    return bool(e._check_success())
                return self.open_fixture(fixture, joint, handle, regrasp + 1)
        else:
            raise RuntimeError('Joint feedback did not converge')
        p, Q = self.pose()
        for _ in range(50):
            self.tick(p, Q, 1)
        self.phase = 'handle_release'
        for _ in range(50):
            self.tick(p, Q, -1)
        self.move(p + out * 0.12, -1, n=70, hold=30, rot=Q)
        return bool(e._check_success())
