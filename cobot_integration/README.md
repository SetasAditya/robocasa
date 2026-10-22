# CoBot task suite

The `cobot-task-suite` branch adds CoBot mobile manipulation and shared
scene preparation and demonstration collection to RoboCasa. It includes a CoBot
mobile base, Kinova arm, linked Robotiq 2F85 gripper, paired simulated cameras,
eight task families, and a browser teleoperation collector.

## What is included

| Component | Purpose |
|---|---|
| `cobot.py`, `cobot_fourbar.py`, `assets/` | Robot registration and the deployed base/gripper geometry, with portable mesh paths. Kinova arm assets come from robosuite. |
| `kitchen_tasks.py` | Shared robot/controller settings and physical surface-transfer and collection-bin task extensions. |
| `kitchen_recording.py`, `moma_validation.py` | Synchronized camera/action/state recording and replay support. |
| `run_scenarios.py` | Run the suite using this checkout and an existing RoboCasa installation. |
| `../cobot_scenarios/` | Task definitions, scene preparation, browser controls, replay, assignments, acceptance and coverage. |
| `PROVENANCE.json`, `source_commits.json` | Deployed source hashes, dependency commits and packaging changes. |
| `VERIFICATION.json` | Checks performed on this branch and outstanding validation. |

The committed curriculum contains **1,000 scenario requests: 900 training and
100 validation**, including **96 human seed slots**. Requests specify the task,
kitchen, objects and variations; they are not completed trajectories. Reserved
test kitchens and object instances are defined separately from these requests.
See the [task and collection guide](../cobot_scenarios/README.md).

## Start on RoboVision

Run from the checkout. The launcher loads this branch's CoBot/task code while
reusing the installed RoboCasa kitchens and simulator. The existing runtime is
left unchanged. Keep generated scenes and recordings outside the repository.

```bash
cd /home/aditya/robocasa-cobot-tasks
export MUJOCO_GL=egl
export COBOT_PY=/robodata/aditya/robocasa_kitchen_aug_v1/runtime/venv/bin/python
export COBOT_NATIVE=/robodata/aditya/robocasa_kitchen_aug_v1/runtime/robocasa
export COBOT_ASSETS="$COBOT_NATIVE/robocasa/models/assets"
export COBOT_RUN="$PWD/cobot_integration/run_scenarios.py"
export COBOT_INDEX="$PWD/cobot_scenarios/data/scenario_index.json"
export COBOT_DATA=/robodata/aditya/robocasa_kitchen_aug_v1/cobot_task_collection
mkdir -p "$COBOT_DATA"

# Validate the committed curriculum; this needs only standard-library Python.
python3 "$COBOT_RUN" check \
  --registry cobot_scenarios/data/registry.json --index "$COBOT_INDEX"

# Prepare a cabinet example. Choose other IDs from the shared index.
$COBOT_PY "$COBOT_RUN" --robocasa-root "$COBOT_NATIVE" prepare \
  --index "$COBOT_INDEX" --id 61f18ef8543651218df8 \
  --assets "$COBOT_ASSETS" --out "$COBOT_DATA/bundles/61f18ef8543651218df8"

# Launch browser collection for the frozen scene.
$COBOT_PY "$COBOT_RUN" --robocasa-root "$COBOT_NATIVE" serve \
  --bundle "$COBOT_DATA/bundles/61f18ef8543651218df8" \
  --recordings "$COBOT_DATA/recordings" --port 8766
```

From a laptop, run `ssh -N -L 8766:127.0.0.1:8766 robovision` and open the
authenticated console URL printed by the server, using localhost port 8766.
Use B to switch base/arm control and hold Shift to enable movement. Reset,
Record, perform the task, then Save. The task guide covers replay and acceptance.

On another machine, install the RoboCasa assets and dependencies identified in
`source_commits.json`, then change the paths above. The bundled robot does not
require a separate RoboLab checkout. Frozen scenes pin their runtime and asset
paths; prepare new bundles when those change.

## Validation and scope

- The branch passes 15 standard-library tests and validates all 1,000 requests.
- The retained physical task classes, nine robot meshes and controller settings
  match the source deployment. Robot XML changes only make mesh paths relative.
- The source deployment's local sample passed initialization for 40/40 scenes;
  37 passed conservative approach screening and three need navigation review.
- A fresh simulator run on RoboVision remains unverified: startup stalled on NFS
  reads before simulation loaded. Successful task trajectories are not certified.

Run the package checks from the repository root:

```bash
PYTHONPATH=. python3 -m unittest discover -s cobot_scenarios/tests -v
```

The collector uses synchronous requests; simulator actions use 20 Hz timing, but
wall-clock streaming speed depends on rendering. Robot-relative VLA formatting
and automatic MimicGen expansion are separate work. Expansion slots specify
coverage goals rather than guaranteed augmentation results.

Generated scenes, recordings, caches, assignment databases and legacy campaign
scripts are excluded. The [robot asset notes](assets/README.md) describe geometry
and licensing; camera calibration is provisional simulation calibration.
