"""Passive stability and visual-bound invariance of LabApt collision repairs."""
import argparse, json
from pathlib import Path
import numpy as np, mujoco
from ._scene import scene
p = argparse.ArgumentParser()
p.add_argument('--output', type=Path, required=True)
p.add_argument('--baseline', type=Path)
a = p.parse_args()
a.output.mkdir(exist_ok=True, parents=True)
scene(a.output, False)
m = mujoco.MjModel.from_xml_path(str(a.output / 'scene.xml'))
d = mujoco.MjData(m)
mujoco.mj_forward(m, d)
visuals = 0
if a.baseline is not None:
    old = mujoco.MjModel.from_xml_path(str(a.baseline))
    od = mujoco.MjData(old)
    mujoco.mj_forward(old, od)
    for i in range(old.ngeom):
        if old.geom_group[i] != 1:
            continue
        name = mujoco.mj_id2name(old, mujoco.mjtObj.mjOBJ_GEOM, i)
        j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, name)
        for before, after in [(old.geom_size[i], m.geom_size[j]), (od.geom_xpos[i], d.geom_xpos[j]), (od.geom_xmat[i], d.geom_xmat[j])]:
            np.testing.assert_allclose(before, after, atol=1e-09, rtol=0, err_msg=name)
        visuals += 1
initial = d.qpos.copy()
maximum = np.zeros(m.nq)
for _ in range(round(25 / m.opt.timestep)):
    mujoco.mj_step(m, d)
    maximum = np.maximum(maximum, np.abs(d.qpos - initial))
tracked = {}
for i in range(m.njnt):
    name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i)
    if any((x in name for x in ('doorhinge', 'door_joint', 'microjoint', 'slidejoint', 'rack'))):
        drift = float(maximum[m.jnt_qposadr[i]])
        tracked[name] = drift
        assert drift < 0.001, (name, drift)
load_checks = {}
for name, torque in [('fridge_labapt_fridge_door_joint', 5.0), ('microwave_labapt_microjoint', -0.5)]:
    test = mujoco.MjData(m)
    jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
    for _ in range(round(5 / m.opt.timestep)):
        test.qfrc_applied[m.jnt_dofadr[jid]] = torque
        mujoco.mj_step(m, test)
    angle = float(test.qpos[m.jnt_qposadr[jid]])
    assert abs(angle) > 1.5, (name, angle)
    load_checks[name] = dict(torque_nm=torque, seconds=5, final_angle_rad=angle)
report = dict(passive_seconds=25, maximum_joint_drift=tracked, unchanged_visual_geometries=visuals, isolated_mechanical_load_checks=load_checks, passed=True)
(a.output / 'dynamics_report.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
