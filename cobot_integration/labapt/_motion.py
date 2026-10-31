"""Action-only motion helpers for optional LabApt probes.

Only the used dependency closure of the deployed controllers is included.
"""
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

class MobileControl:
    def __init__(self, env, step=None):
        self.env = env
        self.robot = env.robots[0]
        self.step = step or env.step
        self.site = self.robot.eef_site_id['right']
        self.steps = 0
        self.phase = 'idle'
        self.rotation = np.diag([1.0, -1.0, -1.0])
        self.grip = -1

    def pose(self):
        d = self.env.sim.data
        return (d.site_xpos[self.site].copy(), d.site_xmat[self.site].reshape(3, 3).copy())

    def base(self):
        return self.env.sim.data.get_body_xpos('mobilebase0_base').copy()

    def tick(self, pos, rot=None, grip=None, base=None):
        if rot is None:
            rot = self.rotation
        if grip is None:
            grip = self.grip
        self.grip = grip
        ac = self.robot.create_action_vector({'right': np.r_[pos, Rotation.from_matrix(rot).as_rotvec()], 'right_gripper': [grip], 'base': np.zeros(3) if base is None else np.asarray(base), 'base_mode': -1})
        out = self.step(ac)
        self.steps += 1
        if not np.isfinite(self.env.sim.data.qpos).all():
            raise RuntimeError('Non-finite simulator state')
        return out

    def move(self, target, grip=None, n=80, hold=25, rot=None):
        target = np.asarray(target)
        p, Q = self.pose()
        R = self.rotation if rot is None else rot
        inter = Slerp([0, 1], Rotation.from_matrix(np.stack([Q, R])))
        for t in np.linspace(0, 1, n):
            self.tick(p + (target - p) * t, inter(t).as_matrix(), grip)
        for _ in range(hold):
            self.tick(target, R, grip)
        return float(np.linalg.norm(self.pose()[0] - target))

    def drive_delta_world(self, delta, grip=None, max_steps=2500, tolerance=0.05, speed=0.1):
        initial = self.base()
        target = initial[:2] + np.asarray(delta)[:2]
        eef, Q = self.pose()
        self.phase = 'loaded_transport' if self.grip > 0 else 'base_approach'
        R = self.env.sim.data.get_body_xmat('mobilebase0_base')[:2, :2].copy()
        for _ in range(max_steps):
            now = self.base()
            err = target - now[:2]
            if np.linalg.norm(err) < tolerance:
                break
            worldvel = np.clip(err * 1.2, -speed, speed)
            c = self.robot.part_controllers['base']
            _, ori = c.get_base_pose()
            theta = np.arctan2(ori[1, 0], ori[0, 0]) - np.arctan2(c.init_ori[1, 0], c.init_ori[0, 0])
            ids = [self.env.sim.model.joint_name2id('mobilebase0_joint_mobile_' + n) for n in ('forward', 'side')]
            axes = np.stack([self.env.sim.data.xaxis[j][:2] for j in ids], axis=1)
            jointvel = np.linalg.solve(axes, worldvel)
            vel = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]) @ jointvel
            self.tick(eef + now - initial, Q, grip, np.r_[vel, 0.0])
        else:
            raise RuntimeError('Kitchen base approach blocked: ' + str(err))
        for _ in range(30):
            self.tick(eef + self.base() - initial, Q, grip)
        return self.base() - initial

    def place(self, target, release_height=0.004):
        target = np.asarray(target, dtype=float)
        self.phase = 'place_approach'
        err = self.move(target + [0, 0, 0.12], 1, n=70)
        print('PLACE_APPROACH', err, self.pose()[0], target, flush=True)
        self.phase = 'place_descend'
        err = self.move(target + [0, 0, release_height], 1, n=70, hold=35)
        print('PLACE_DESCEND', err, self.pose()[0], target, flush=True)
        self.phase = 'release'
        p = self.pose()[0].copy()
        for _ in range(80):
            self.tick(p, grip=-1)
        print('RELEASE_QPOS', [float(self.env.sim.data.get_joint_qpos(j)) for j in self.robot.gripper['right'].joints], flush=True)
        self.phase = 'retreat'
        self.move(p + [0, 0, 0.16], -1, n=70, hold=30)


class PosturePlanner:
    def __init__(self, expert):
        self.expert = expert
        self.env = expert.env

    def align_pose_via_fk(self, target, R):
        """Plan in an independent MjData clone; execute ordinary OSC actions only."""
        import mujoco
        from scipy.optimize import least_squares
        from scipy.spatial.transform import Rotation
        env = self.env
        x = self.expert
        model = env.sim.model._model
        clone = mujoco.MjData(model)
        clone.qpos[:] = env.sim.data.qpos
        clone.qvel[:] = 0
        ids = [env.sim.model.joint_name2id(f'robot0_Actuator{i}') for i in range(1, 8)]
        addresses = [int(model.jnt_qposadr[i]) for i in ids]
        initial = clone.qpos[addresses].copy()
        low = []
        high = []
        for jid, q in zip(ids, initial):
            a, b = model.jnt_range[jid] if model.jnt_limited[jid] else (q - 2 * np.pi, q + 2 * np.pi)
            low.append(a + 0.005)
            high.append(b - 0.005)
        margin = float(getattr(self, 'joint6_margin', 0.0))
        if margin < 0 or low[5] + margin >= high[5] - margin:
            raise ValueError('Invalid planner-only joint6 margin')
        low[5] += margin
        high[5] -= margin
        if not getattr(self, 'free_elbow', False):
            high[3] = min(high[3], -0.15)

        def forward(q):
            clone.qpos[addresses] = q
            mujoco.mj_forward(model, clone)
            return (clone.site_xpos[x.site].copy(), clone.site_xmat[x.site].reshape(3, 3).copy())

        def residual(q):
            position, rotation = forward(q)
            error = np.r_[(position - target) * 5.0, Rotation.from_matrix(rotation.T @ R).as_rotvec()]
            if getattr(self, 'free_elbow', False):
                error = np.r_[error, 0.01 * (q[1] - 1.0), 0.01 * (q[3] - 1.8)]
            return error
        candidates = []
        for seed in [initial, np.array([0.0, 1.0, np.pi, -1.8, 0.0, 1.0, np.pi / 2]), np.array([0.0, 0.3, np.pi, -2.3, 0.0, 1.5, 0.0]), np.array([0.0, 0.8, np.pi, 1.8, 0.0, -1.0, 0.0]), np.array([1.0, 1.0, 0.0, 1.8, 1.0, 1.0, 2.0])]:
            result = least_squares(residual, np.clip(seed, np.array(low) + 1e-06, np.array(high) - 1e-06), bounds=(low, high), max_nfev=180)
            candidates.append(result)

        def fixture_collision(q):
            forward(q)
            for contact in clone.contact:
                names = [env.sim.model.geom_id2name(int(g)) or '' for g in (contact.geom1, contact.geom2)]
                robot = [n.startswith(('robot0_', 'gripper0_')) for n in names]
                if sum(robot) == 2:
                    if contact.dist < -0.003:
                        return (names, float(contact.dist))
                    continue
                if sum(robot) != 1:
                    continue
                other = names[0 if robot[1] else 1]
                mast = other.startswith('mobilebase0_') and ('sensor_mast' in other or 'sensor_top_bar' in other)
                if contact.dist >= (-0.0001 if mast else -0.003):
                    continue
                if other.startswith('mobilebase0_') and (not mast) or 'floor' in other or 'ground' in other:
                    continue
                return (names, float(contact.dist))
            return None
        ranked = []
        diagnostics = []
        endpoint_goals = []
        for result in candidates:
            planned = result.x.copy()
            for k, jid in enumerate(ids):
                if not model.jnt_limited[jid]:
                    planned[k] += 2 * np.pi * np.round((initial[k] - planned[k]) / (2 * np.pi))
            position, rotation = forward(planned)
            error = float(np.linalg.norm(position - target))
            orientation = float(np.linalg.norm(Rotation.from_matrix(rotation.T @ R).as_rotvec()))
            collision = None
            endpoint_collision = fixture_collision(planned) if error <= 0.025 and orientation <= 0.12 else None
            if error <= 0.025 and orientation <= 0.12:
                for t in np.linspace(0, 1, 20):
                    collision = fixture_collision(initial + (planned - initial) * t)
                    if collision is not None:
                        break
            if error <= 0.025 and orientation <= 0.12 and (endpoint_collision is None):
                endpoint_goals.append(planned)
            feasible = error <= 0.025 and orientation <= 0.12 and (collision is None)
            diagnostics.append(dict(position_error=error, orientation_error=orientation, collision=collision, endpoint_collision=endpoint_collision, feasible=feasible, joint_distance=float(np.linalg.norm(planned - initial))))
            if feasible:
                ranked.append((float(np.linalg.norm(planned - initial)), planned, error, orientation))
        joint_path = None
        if not ranked and getattr(self, 'use_rrt_connect', False) and endpoint_goals:
            from ._rrt import plan as rrt_connect_plan
            candidate_path = rrt_connect_plan(initial, endpoint_goals, np.asarray(low), np.asarray(high), fixture_collision)
            if candidate_path is not None:
                joint_path = candidate_path
                planned = np.asarray(joint_path[-1]).copy()
                position, rotation = forward(planned)
                path_length = sum((np.linalg.norm(b - a) for a, b in zip(joint_path, joint_path[1:])))
                ranked.append((float(path_length), planned, float(np.linalg.norm(position - target)), float(np.linalg.norm(Rotation.from_matrix(rotation.T @ R).as_rotvec()))))
                print('CLONE_RRT_CONNECT', dict(waypoints=len(joint_path), path_length=float(path_length)), flush=True)
        if not ranked and getattr(self, 'use_rrt', False) and endpoint_goals:
            rng = np.random.default_rng(1729)
            nodes = [initial.copy()]
            parents = [-1]

            def edge_free(a, b):
                count = max(2, int(np.ceil(np.linalg.norm(b - a) / 0.12)) + 1)
                return all((fixture_collision(a + (b - a) * t) is None for t in np.linspace(0, 1, count)))
            for iteration in range(900):
                goal = endpoint_goals[iteration % len(endpoint_goals)]
                sample_q = goal if rng.random() < 0.25 else np.clip(initial + rng.random() * (goal - initial) + rng.normal(0, 1.0, 7), low, high)
                index = int(np.argmin([np.linalg.norm(q - sample_q) for q in nodes]))
                delta = sample_q - nodes[index]
                next_q = nodes[index] + delta * min(1.0, 0.4 / max(np.linalg.norm(delta), 1e-09))
                if not edge_free(nodes[index], next_q):
                    continue
                nodes.append(next_q)
                parents.append(index)
                if not edge_free(next_q, goal):
                    continue
                joint_path = [goal.copy(), next_q.copy()]
                index = parents[-1]
                while index >= 0:
                    joint_path.append(nodes[index].copy())
                    index = parents[index]
                joint_path.reverse()
                planned = goal.copy()
                position, rotation = forward(planned)
                ranked.append((sum((np.linalg.norm(b - a) for a, b in zip(joint_path, joint_path[1:]))), planned, float(np.linalg.norm(position - target)), 0.0))
                print('CLONE_RRT', dict(iterations=iteration + 1, nodes=len(nodes), waypoints=len(joint_path)), flush=True)
                break
        if not ranked:
            print('CLONE_IK_REJECTED', diagnostics, flush=True)
            print('CLONE_IK_REJECT', diagnostics, flush=True)
            raise RuntimeError('No pose-feasible IK candidate has a collision-free 20-sample joint interpolation')
        _, planned, error, orientation = min(ranked, key=lambda item: item[0])
        print('CLONE_IK', dict(position_error=error, orientation_error=orientation, planned_qpos=planned.tolist(), candidate_diagnostics=diagnostics), flush=True)
        x.phase = 'cabinet_posture_waypoints'
        if joint_path is None:
            joint_path = [initial, planned]
        length = sum((np.linalg.norm(b - a) for a, b in zip(joint_path, joint_path[1:])))
        for qa, qb in zip(joint_path, joint_path[1:]):
            for fraction in np.linspace(0, 1, max(20, int(320 * np.linalg.norm(qb - qa) / max(length, 1e-09)))):
                position, rotation = forward(qa + (qb - qa) * fraction)
                x.tick(position, rotation, -1)
        for _ in range(60):
            x.tick(target, R, -1)
        actual_error = float(np.linalg.norm(x.pose()[0] - target))
        if actual_error > 0.04:
            raise RuntimeError(f'FK waypoint execution failed physical tracking: {actual_error:.3f}m')
        return dict(planned_qpos=planned.tolist(), final_position_error=actual_error, candidate_diagnostics=diagnostics)
