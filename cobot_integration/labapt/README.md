# LabApt kitchen

LabApt is RoboCasa layout `100`, named `LayoutType.LAYOUT100` or
`LayoutType.LAB_APT`. Its blueprint lives beside the other held-out layouts at
`robocasa/models/assets/scenes/kitchen_layouts/test/layout100.yaml`.

Use normal RoboCasa environment construction with `layout_ids=[100]`, or the
existing Cobot factory:

```python
import sys
sys.path.insert(0, "cobot_integration")
from kitchen_tasks import make_env

env = make_env("OpenMicrowave", seed=0, layout="LAB_APT", style=1)
obs = env.reset()
# Run normal robot actions.
env.close()
```

For script-style Cobot imports, add `cobot_integration` to `PYTHONPATH`, as the
existing Cobot launch scripts do. Native `robosuite.make` construction requires
no LabApt tool imports.

The measured layout, fixture poses, dimensions, and appearance are fixed.
Native tasks still sample movable objects, placements, and initial states.
Choose LabApt explicitly: it is excluded from the standard TRAIN, TEST, and ALL
groups to keep it held out from training. Style IDs preserve the lab appearance.

## Optional tools

Run from this checkout in the environment containing RoboCasa/robosuite dependencies.
The native kitchen/object assets must be installed for this checkout, as for other layouts.
The original scans and archived development scripts are not required:

```bash
python -m cobot_integration.labapt validate --output /tmp/labapt_checks
python -m cobot_integration.labapt render --output /tmp/labapt_views
python -m cobot_integration.labapt inventory --output /tmp/labapt_tasks.json
python -m cobot_integration.labapt resets --output /tmp/labapt_resets --jobs 8
python -m cobot_integration.labapt rollouts --output /tmp/labapt_rollouts --seeds 3 --jobs 4
```

Output directories must be new, except the resumable `resets` sweep. Each command
accepts `--help`. Rendering uses headless EGL. Validation checks metric geometry,
existing layouts, passive stability, and isolated appliance actuation. The rollout
tools use privileged scripted controllers and retain action/state evidence and
contact audits. They do not qualify paired-camera training data or real-world
transfer. Normal task use does not require these tools or the original scans.

## Geometry and task support

The adjacent `layout100.manifest.json` records the scan transform and fixture
measurements; `layout100.walls.json` records fitted wall planes, openings, and
uncertainty. `layout100.fixture_baseline.json` preserves the original fixture
records. Coordinates are meters, Z up. Episode metadata includes the calibration
and asset hashes. ROS map-to-scan and real camera calibration remain separate.

The fitted fridge and microwave retain source visuals and inferred dynamics.
Hidden collision proxies include the documented v4 clearances: microwave door
panel offset, cabinet/drawer panel clearance, and a 4% width reduction of fridge
door-storage collision meshes. These corrections preserve visible dimensions;
their physical approximations still require real-world measurement.

The kitchen has base cabinets, pantry, drawers, fridge, microwave, dishwasher,
sink, shelves, and a dining table. The air fryer is a fixed obstacle. Native tasks
may enable the scan-positioned dish rack. No oven, stove, coffee machine, or
fridge drawer has been fabricated. Task compatibility depends on installed object
assets, available space, and required affordances.

The v4 evaluation passed cabinet, drawer, fridge, and microwave opening, plus
table-to-counter transfer, on three seeds each, with action replay and recorded
contact checks. That evidence covers these five workflows. The broader inventory
contains 112 native configurations with usable reset recipes; it does not certify
physical completion of the entire task catalog.

Historical authoring tools, versioned reports, and patch baselines are preserved
outside the runtime checkout under
`/robodata/aditya/robocasa_kitchen_aug_v1/labapt_cleanup_v5/` and the sibling
`labapt_validation_*` / `labapt_fixes_v4` directories.

The shared factory accepts native environment names plus `fetch_surface`,
`bin_placement`, and `collect_two_stations`. Other compositional task families
continue to use the existing `cobot_scenarios` registry and scene preparation.
For LabApt, load this branch's RoboCasa source; an explicit `--robocasa-root`
selects that other source tree and requires the layout to be installed there.
