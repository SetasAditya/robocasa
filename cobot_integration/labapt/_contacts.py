"""Sample recorded-state contacts; diagnostic reconstruction, not action replay."""
import argparse, json
from pathlib import Path
import h5py, mujoco, numpy as np
p = argparse.ArgumentParser()
p.add_argument('dataset', type=Path)
p.add_argument('--stride', type=int, default=10)
a = p.parse_args()
with h5py.File(a.dataset) as f:
    g = f['data/demo_0']
    model = mujoco.MjModel.from_xml_string(g.attrs['model_file'])
    states = g['states'][::a.stride]
    final = g['final_state'][:]
    phases = g['phases'].asstr()[::a.stride]
data = mujoco.MjData(model)
names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or '' for i in range(model.ngeom)]
worst = {}
counts = 0
for k, state in enumerate(np.vstack([states, final])):
    data.qpos[:] = state[1:1 + model.nq]
    mujoco.mj_forward(model, data)
    for c in data.contact:
        pair = [names[int(i)] for i in (c.geom1, c.geom2)]
        robot = [n.startswith(('robot0_', 'gripper0_', 'mobilebase0_')) for n in pair]
        if not any(robot) or any(('floor' in n or 'ground' in n for n in pair)):
            continue
        if c.dist >= -0.005:
            continue
        counts += 1
        key = ' / '.join(pair)
        if key not in worst or worst[key]['depth_m'] < -c.dist:
            worst[key] = dict(depth_m=float(-c.dist), step=k * a.stride, phase=str(phases[k]) if k < len(phases) else 'final')
r = dict(dataset=str(a.dataset), stride=a.stride, penetrations_over_5mm=worst, sampled_contact_count_over_5mm=counts)
a.dataset.with_name('contact_audit.json').write_text(json.dumps(r, indent=2) + '\n')
print(json.dumps(r, indent=2))
