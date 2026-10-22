"""Explicit simulator smoke: never adds its unsuccessful recording to the dataset."""
import argparse,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--integration',required=True);p.add_argument('--bundle',required=True);p.add_argument('--out',required=True);a=p.parse_args()
from cobot_scenarios.runtime import bootstrap,restore
bootstrap(a.integration)
from cobot_scenarios.browser import Session,replay
import numpy as np
session=Session(a.bundle,a.out)
initial=session.env.sim.get_state().flatten().copy();session.reset()
assert np.array_equal(initial,session.env.sim.get_state().flatten()),'Reset changed initial state'
status=session.status();assert all(status['images'].values())
session.start();env=session.env;sid=env.robots[0].eef_site_id['right'];tcp=env.sim.data.site_xpos[sid].copy();base=env.sim.data.get_body_xpos('mobilebase0_base').copy()
for i in range(12):session.step(dict(mode='arm',move=[0,0,1],stage='approach'))
arm_motion=float(np.linalg.norm(env.sim.data.site_xpos[sid]-tcp))
for i in range(12):session.step(dict(mode='base',move=[1,0,0],stage='approach'))
base_motion=float(np.linalg.norm(env.sim.data.get_body_xpos('mobilebase0_base')-base))
for i in range(6):session.step(dict(mode='arm',turn=[0,0,1],closed=True,stage='approach'))
path=session.save();env.close()
assert arm_motion>.001,arm_motion
assert base_motion>.001,base_motion
result=replay(a.bundle,path)
assert result['replay'],result
assert result['sensor_alignment'],result
result.update(arm_motion_m=arm_motion,base_motion_m=base_motion,recording=path,reset_exact=True)
Path(a.out,'smoke_report.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
