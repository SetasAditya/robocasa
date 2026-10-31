"""Experimental LabApt controllers. Geometry read for planning; execution uses actions only."""
import heapq
import numpy as np
from scipy.spatial.transform import Rotation
from ._motion import MobileControl
from ._motion import PosturePlanner

class LabAptExpert(MobileControl):

    def posture(self, target, R):
        planner = PosturePlanner(self)
        planner.free_elbow = True
        planner.use_rrt_connect = True
        return planner.align_pose_via_fk(np.asarray(target), R)

    def navigate(self, target):
        target = np.asarray(target)[:2]
        start = self.base()[:2].copy()
        obstacles = []
        for f in self.env.fixtures.values():
            if f.name.endswith('_room') or f.name.startswith(('wall_shelf', 'sink_labapt', 'dish_rack')):
                continue
            try:
                center = self.env.sim.data.get_body_xpos(f.root_body)[:2]
                R = self.env.sim.data.get_body_xmat(f.root_body)[:2, :2]
                half = np.asarray(f.size[:2]) / 2
            except Exception:
                continue
            obstacles.append((center.copy(), R.copy(), half + 0.26))
        start_pen = [float(np.min(half - np.abs(R.T @ (start - center)))) for center, R, half in obstacles]

        def free(p):
            for (center, R, half), initial in zip(obstacles, start_pen):
                penetration = float(np.min(half - np.abs(R.T @ (p - center))))
                if penetration > 0 and (not (initial > 0 and np.linalg.norm(p - start) < 0.6 and (penetration <= initial + 0.001))):
                    return False
            return True
        spacing = 0.08
        origin = np.array([-3.0, -6.0])
        shape = (101, 88)
        cell = lambda p: tuple(np.rint((p - origin) / spacing).astype(int))
        point = lambda c: origin + spacing * np.asarray(c)
        first, last = (cell(start), cell(target))
        if not free(target):
            raise RuntimeError('Requested base goal overlaps inflated fixture footprint')
        frontier = [(0.0, first)]
        cost = {first: 0.0}
        parent = {}
        found = False
        while frontier:
            _, node = heapq.heappop(frontier)
            if node == last:
                found = True
                break
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)):
                nxt = (node[0] + dx, node[1] + dy)
                if not (0 <= nxt[0] < shape[0] and 0 <= nxt[1] < shape[1]):
                    continue
                if not free(point(nxt)):
                    continue
                if dx and dy and (not free(point((node[0] + dx, node[1]))) or not free(point((node[0], node[1] + dy)))):
                    continue
                new = cost[node] + np.hypot(dx, dy)
                if new >= cost.get(nxt, float('inf')):
                    continue
                cost[nxt] = new
                parent[nxt] = node
                heapq.heappush(frontier, (new + np.linalg.norm(np.array(nxt) - last), nxt))
        if not found:
            raise RuntimeError('No footprint-clear LabApt base route')
        path = [target]
        node = last
        while node != first:
            path.append(point(node))
            node = parent[node]
        path.append(start)
        path = path[::-1]
        route = []
        index = 0
        while index < len(path) - 1:
            end = index + 1
            for j in range(index + 2, len(path)):
                if all((free(p) for p in np.linspace(path[index], path[j], max(2, int(np.linalg.norm(path[j] - path[index]) / 0.04) + 1)))):
                    end = j
                else:
                    break
            route.append(path[end])
            index = end
        self.last_route = [p.tolist() for p in route]
        for p in route:
            self.drive_delta_world(p - self.base()[:2], self.grip, max_steps=max(700, int(np.linalg.norm(p - self.base()[:2]) / 0.0015) + 250), tolerance=0.035, speed=0.1)
        return route

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
        self.posture(h + out * 0.16, R)
        self.phase = 'handle_engage'
        err = self.move(h, -1, n=110, hold=45, rot=R)
        if err > 0.035:
            raise RuntimeError(f'LabApt handle tracking error {err:.3f}m')
        self.phase = 'handle_close'
        for _ in range(100):
            self.tick(h, R, 1)
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

    def fetch(self):
        e = self.env
        name = 'obj'
        obj = e.objects[name]
        bid = e.obj_body_id[name]
        groups = [self.robot.gripper['right'].important_geoms[k] for k in ('left_finger', 'right_finger')]
        source = e.sim.data.body_xpos[bid].copy()
        Rf = e.sim.data.get_body_xmat(e.source.root_body)
        out = Rf @ np.array([0.0, -1.0, 0.0])
        self.phase = 'fetch_base_approach'
        self.navigate(source[:2] + out[:2] * 0.42)
        R = Rf @ np.diag([1.0, -1.0, -1.0])
        self.rotation = R
        self.phase = 'fetch_pregrasp'
        self.posture(source + [0, 0, 0.18], R)
        self.phase = 'fetch_descend'
        error = self.move(source, -1, n=100, hold=40, rot=R)
        if error > 0.025:
            raise RuntimeError(f'Object approach error {error:.3f}m')
        self.phase = 'fetch_grasp'
        p, Q = self.pose()
        for _ in range(100):
            self.tick(p, Q, 1)
        if not e._check_grasp(groups, obj):
            raise RuntimeError('No bilateral object grasp')
        self.phase = 'fetch_lift'
        self.move(p + [0, 0, 0.22], 1, n=140, hold=40, rot=Q)
        if not e._check_grasp(groups, obj):
            raise RuntimeError('Object grasp lost during lift')
        offset = self.pose()[0] - e.sim.data.body_xpos[bid]
        regions = e.target.get_reset_regions(e)
        candidates = []
        Rt = e.sim.data.get_body_xmat(e.target.root_body)
        for region in regions.values():
            if min(region['size'][:2]) < 0.12:
                continue
            p = np.asarray(e.target.pos) + Rt @ np.asarray(region['offset'])
            p[2] = float(e.target.pos[2] + e.target.size[2] / 2) + float(source[2] - (e.source.pos[2] + e.source.size[2] / 2))
            candidates.append(p)
        if not candidates:
            raise RuntimeError('No clear native destination region')
        target = min(candidates, key=lambda p: np.linalg.norm(p[:2] - self.base()[:2]))
        outward = Rt @ np.array([0.0, -1.0, 0.0])
        base_target = target[:2] + outward[:2] * 0.65
        self.phase = 'loaded_transport'
        self.navigate(base_target)
        if not e._check_grasp(groups, obj):
            raise RuntimeError('Object grasp lost during navigation')
        self.rotation = Q
        self.place(target + offset)
        self.phase = 'fetch_clear'
        p, Q = self.pose()
        self.move(p + np.array([0.0, 0.0, 0.15]), -1, n=70, hold=30, rot=Q)
        p, Q = self.pose()
        self.posture(p + np.array([0.25, 0.0, 0.0]), Q)
        self.phase = 'fetch_clear'
        return dict(native_success=bool(e._check_success()), base_target=base_target.tolist(), object_target=target.tolist(), route=self.last_route)
