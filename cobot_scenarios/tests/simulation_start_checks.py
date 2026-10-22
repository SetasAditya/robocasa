"""Verify rejected starts fail closed without changing the saved simulator state."""
import argparse
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
from cobot_scenarios import runtime

p=argparse.ArgumentParser()
p.add_argument('--integration',required=True)
p.add_argument('--bundle',required=True)
p.add_argument('--out',required=True)
a=p.parse_args()
runtime.bootstrap(a.integration)
env,resolved=runtime.restore(a.bundle,False)
try:
    initial=env.sim.get_state().flatten().copy()
    checks={}
    for name,bounds,contacts in (
        ('no_footprint_clear_candidate',[(-100,-100,100,100)],[]),
        ('all_full_robot_candidates_collide',[],[{'penetration_m':.1}]),
    ):
        with patch('cobot_scenarios.geometry.fixture_bounds',return_value=bounds),patch.object(runtime,'collisions',return_value=contacts):
            try:
                runtime.apply_start(env,resolved['request'])
            except ValueError as error:
                assert 'No collision-free start' in str(error),error
            else:
                raise AssertionError('Unsafe start was accepted')
        assert np.array_equal(initial,env.sim.get_state().flatten()),name
        checks[name]='passed; original state restored exactly'
    Path(a.out).write_text(json.dumps(checks,indent=2))
    print(json.dumps(checks))
finally:
    env.close()
