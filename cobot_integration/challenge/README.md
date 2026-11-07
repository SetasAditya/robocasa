# LabApt manipulation challenge v1

Ten custom task profiles, two overlapping families, three difficulty levels and
two initialization modes. The kitchen remains measured layout 100. Existing
native tasks, layout groups, runtime pins and historical scores are unchanged.

Every profile is **experimental** until its individual scenes pass qualification.
Task implementation, valid resets, witness success, policy evaluation and physical
validation are separate evidence. The suite does not assume learning will win.

## Commands

Run with the RoboCasa dependency interpreter from this checkout:

```bash
python -m unittest discover -s cobot_integration/challenge/tests
python -m cobot_integration.challenge plan --output /new/path/plan.json
python -m cobot_integration.challenge prepare \
  --profile ShallowDrawerRetrieval --difficulty easy --mode workflow \
  --scene /new/path/drawer-scene
python -m cobot_integration.challenge qualify --scene /path/drawer-scene \
  --witness /path/successful-actions.npy --attestations /path/review.json
python -m cobot_integration.challenge evaluate --scene /path/drawer-scene \
  --controller my_policy_adapter --output /new/path/result
```

`plan` reserves disjoint development, practice and formal seed spaces. Defaults
are 20 practice and 50 formal scenes per task/difficulty/mode: 4,200 evaluation
and practice scenes across the 60 cells. Planning does not build or qualify them.
Use seeds from the plan when constructing campaign scenes. All methods reuse the
same frozen formal scenes. Do not choose scenes using RAX performance.

`prepare` refuses overwrite. Recordings and frozen model XML remain outside git.
A scene stores its recipe, source digest, native sampled-object metadata, source
model XML, physical state, temporal scoring state, initialization actions, physical gripper command state and
physical parameters. External mesh/texture files are also hashed. Editing any frozen artifact invalidates its digest.

Manipulation-only cells require `--initialization actions.npy`: recorded robot
actions from the ordinary reset to the approach checkpoint. Recovery cells also
require a physically sustained grasp and reviewed `nominal_grasp_tool_xyz` geometry;
the measured offset must match the selected difficulty within 5 mm. Never attach the object or teleport it into
the gripper. Geometry review verifies the start pose and intended grasp offset.

Rack tasks require `--geometry geometry.json` with `slot.center` and
`slot.half_extents` in the moving rack body frame, plus `slot.plate_normal_axis` for
plate insertion. Bowl tasks require an asset-specific
`seated_height_interval_m` based on collision geometry and `relationship: stack` or `relationship: nest`: stacking on an inverted support bowl
and nesting inside the cavity must have distinct intervals. These values are
operator truth, not controller inputs. Without them the relevant goal stays false.

## Qualification

A scene needs initially false success, no gross initial penetrations, nonempty
Kinect and wrist renders, stable passive settling, physical state restoration,
and a successful witness action replay twice with matching states and scoring
history. Protected-object motion and object penetrations are checked every control
step. The 5 mm screen is only a gross physics defect threshold, not insertion
clearance validation.

The review JSON must identify `scene_sha256` (SHA-256 of `scene.json`), a nonempty
`reviewer`, and true `collision_geometry_reviewed`, `primary_axis_verified`,
`initialization_verified`. Verify reachable starts, actual object/rack dimensions,
fine clearances, sampled difficulty, support surfaces and task feasibility before
signing. Review is a recorded human/engineering judgment, not automated evidence
of real-world fidelity. Do not sign a profile solely because reset succeeds.

No scene enters formal evaluation until all checks pass. A `--diagnostic` evaluation
can collect controller actions for development or witnesses; its results remain
explicitly labelled diagnostic. To refresh qualification, retain original evidence
externally and supply the new witness/review explicitly.

## Policy interface

`--controller MODULE` imports a trusted adapter exposing:

```python
def create(*, instruction, action_contract):
    return policy  # policy.act(observation) returns one 20 Hz robot action
```

The action contract includes the common CoBot action slices and normalized base-frame delta OSC
arm commands (kp 300). This is an explicit common challenge action contract,
separate from the historical RAX absolute-target tool interface. Exported controllers
must adapt their commands to this contract; historical scores are not comparable. All three controller categories use the same boundary: exported RAX
controllers, engineered feedback controllers and learned skills. Observation
contains rendered Kinect RGB/depth, wrist RGB, calibration, proprioception and
tool pose. No object poses, slot bounds, success state or fixture truth is passed.
No new force sensor is invented. This interface is trusted evaluation code, not
an OS sandbox: run untrusted generated programs through the existing isolated
RAX tool service, not as Python plugins. Wiring or running a RAX learning campaign
and training learned skills are outside this release.

Report success by task, difficulty and mode; also report attempts, completion time,
collisions, drops and recovery. Keep diagnostic, qualified formal and physical
robot results separate. Report pretraining/demonstration/development resources.

## Experimental apparatus

Guided tray insertion adds removable blue countertop guides, without modifying
measured fixtures. Tray geometry and guide fit need review. Sponge wiping measures
physical contact coverage over a 10×10 marked patch, followed by release and a
stable rest. It does not simulate cleaning chemistry or a deformable sponge.

Default numeric difficulty values are development recipes, not calibrated claims
about real-world tolerances. Changing them requires a new suite version and frozen
scenes. Rack, bowls and contact apparatus need finer collision review than the old
reset suite. No real-world validation or controller success is claimed by these
sources.

## Reproduce development checks

```bash
python -m cobot_integration.challenge.tests.runtime_smoke \
  --output /new/path/runtime-checks --jobs 3
```

This bounded check builds one easy workflow seed for each ordinary-reset profile,
renders reset illustrations, checks passive stability and exercises independent
action replay on the drawer case. It deliberately leaves recovery pending a real
recorded grasp. Passing these checks does not provide successful task witnesses
or qualify medium/hard recipes, manipulation checkpoints or real robot execution.
See `VALIDATION.json` for the exact release scope and outstanding evidence.
