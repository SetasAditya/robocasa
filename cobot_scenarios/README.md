# CoBot scenarios and demonstration collection

This package defines shared mobile manipulation tasks for RoboCasa and CoBot.
It prepares reproducible scenes, provides browser teleoperation, saves and
replays demonstrations, and tracks collaborator assignments and accepted coverage.
Start with the [CoBot setup guide](../cobot_integration/README.md) to configure
the RoboVision paths used in the examples below.

## Tasks, scenes and variations

| Task family | Train | Validation | Manipulation goal |
|---|---:|---:|---|
| Surface delivery | 162 | 18 | Move an object between separated counters. |
| Selective delivery | 126 | 14 | Select the instructed object among distractors and deliver it. |
| Dishes to sink | 126 | 14 | Move a dish from a counter into the sink. |
| Cabinet articulation | 90 | 10 | Open or close a cabinet. |
| Drawer articulation | 90 | 10 | Open or close a drawer. |
| Cabinet transfer | 144 | 16 | Open, store/retrieve an object, and close. |
| Microwave transfer | 90 | 10 | Open, load/unload an object, and close. |
| Collect two stations | 72 | 8 | Bring objects from two work surfaces into a physical bin. |
| **Total** | **900** | **100** | |

The 1,000 requests include 96 human seed slots. The remaining requests are
expansion candidates; the package does not guarantee that MimicGen can generate
them. All task families require physical task success and at least 0.3 m of base
displacement. That displacement gate is a minimal mobility requirement, not proof
of meaningful navigation difficulty. Transfer tasks also check opening, transfer
and final closure history. Surface transfers preserve distractor positions;
collection requires both objects to contact the physical bin floor.

Training uses layouts 11/16/24/38 and styles 11/22; validation uses layout 45 and
styles 33/44. Layouts 8/9/10 and styles 3/8 are reserved for testing, with no test
requests generated in this curriculum. Object instances are separated by split
and pinned by XML hashes. Categories include cans, mugs, glass cups, bowls,
plates and cereal packages, depending on the task.

Variations cover object instances and placement/yaw, clutter, lighting, initial
door/drawer state, base heading and base-distance bands of 0.5–0.8, 0.8–1.5 and
1.5–3.0 m. Physics uses nominal settings. `spec.py` defines the task and scene
schedule; `data/registry.json` pins assets and `data/scenario_index.json` contains
the shared requests. Use these committed definitions for collection.

## Scripts and commands

| File | Responsibility |
|---|---|
| `spec.py` | Task quotas, scene splits, variation ranges and success requirements. |
| `core.py` | Asset inventory, deterministic requests, integrity checks, assignments, acceptance and coverage. |
| `runtime.py` | Robot/task registration, scene preparation, initial contact checks, frozen state and exact restore. |
| `geometry.py` | Object-fit checks and conservative base-approach screening. |
| `browser.py`, `console.html` | Browser controls, recording, reset/save and replay. |
| `reporting.py` | Scene preview gallery. |
| `__main__.py` | Command-line entry point. |
| `tests/` | Curriculum, geometry, portability and simulator collection checks. |

`inventory` and `generate` create a new curriculum when deliberately changing
assets or requirements. Write it into a new directory; existing request files are
immutable. `generate --index-only` avoids creating 1,000 separate files on NFS.
Registry, ledger and coverage commands do not need the simulator.

## Prepare a scene and inspect it

Choose a scenario ID from `$COBOT_INDEX`. The examples use the setup variables in
the CoBot guide; replace `SCENARIO_ID` with an actual ID.

```bash
$COBOT_PY "$COBOT_RUN" --robocasa-root "$COBOT_NATIVE" prepare \
  --index "$COBOT_INDEX" --id SCENARIO_ID --assets "$COBOT_ASSETS" \
  --out "$COBOT_DATA/bundles/SCENARIO_ID"
$COBOT_PY "$COBOT_RUN" --robocasa-root "$COBOT_NATIVE" restore-check \
  --bundle "$COBOT_DATA/bundles/SCENARIO_ID"
python3 "$COBOT_RUN" gallery --bundles "$COBOT_DATA/bundles" \
  --out "$COBOT_DATA/gallery"
```

`prepare` resolves fixtures and objects, then saves `model.xml`, `state.npy`,
`resolved.json` and Kinect/wrist previews. The resolved metadata includes the
instruction, controller configuration, source/runtime hashes, object dimensions,
realized sampling and qualification status. Share the entire bundle for exact
restore. Matching runtime and asset mount paths are required.

Object sampling checks the rotated microwave retrieval object's bounds against
the usable region. It records any yaw adjustment while preserving the assigned
asset, scale, fixture and split; impossible fits remain errors. Background clutter
for cabinet and microwave transfers is placed on the task counter. Base sampling
preserves requested distance and heading, searches angular positions, and rejects
candidates that fail footprint or full-robot contact checks. Failed searches
restore the native pose and stop instead of using an unchecked fallback.

Inspect the previews and qualification results before collection. Initialization
and conservative approach screening do not establish arm reach, grasp fit, final
docking or manipulation success. Failed initialization remains visible in
`failure.json`. Use new output directories after changing source or runtime;
frozen bundles deliberately reject a mismatched runtime.

To prepare a sample across the families, use `prepare-batch` with the same index,
assets and an output root, plus `--limit 32`. It records blocked requests rather
than replacing them with easier tasks.

## Collect and replay

```bash
$COBOT_PY "$COBOT_RUN" --robocasa-root "$COBOT_NATIVE" serve \
  --bundle "$COBOT_DATA/bundles/SCENARIO_ID" \
  --recordings "$COBOT_DATA/recordings" --port 8766
```

Connect through the SSH tunnel described in the setup guide. Follow the on-page
key legend: B switches base/arm, and Shift enables movement. Select the task stage,
then Reset, Record, perform the task, and Save. Preview movement requires Reset
before recording. Discard marks the partial recording and restores the scene.

Saved episodes contain raw 11D absolute-world TCP/controller actions, synchronized
RGB and simulator state, task stages, scenario identity and offline base/TCP
transforms. The interface translates relative commands into that action contract.
Simulator steps use 20 Hz timing; rendering limits wall-clock speed. The collector
is synchronous, with no queued key commands. Offline geometry is not a policy
observation, and model-specific VLA conversion remains a separate training step.

```bash
$COBOT_PY "$COBOT_RUN" --robocasa-root "$COBOT_NATIVE" replay \
  --bundle "$COBOT_DATA/bundles/SCENARIO_ID" --recording /path/to/demo.hdf5
```

## Coordinate collection and acceptance

Initialize a shared ledger, then claim a scenario before collecting it:

```bash
python3 "$COBOT_RUN" ledger --db "$COBOT_DATA/collection.sqlite" \
  --index "$COBOT_INDEX"
python3 "$COBOT_RUN" ledger --db "$COBOT_DATA/collection.sqlite" \
  --id SCENARIO_ID --owner alice --revision 0 --status assigned
python3 "$COBOT_RUN" ledger --db "$COBOT_DATA/collection.sqlite" \
  --id SCENARIO_ID --owner alice --revision 1 --status recorded \
  --recording /path/to/demo.hdf5
```

Keep one authoritative server ledger. Owners and revision checks reject conflicting
updates. Offline collaborators can export events with `--export alice-events.json`
and merge them with `--merge alice-events.json` after agreeing assignments.
Identical imported events are idempotent.

Acceptance requires matching scenario identity, source lineage and reviewed
boolean evidence for `physical_success`, `replay`, `sensor_alignment`,
`collision_review` and `instruction_review`. Submit that JSON with
`--status accepted --revision 2 --lineage SOURCE_ID --evidence evidence.json`.
Replay alone does not certify collision or instruction review. Cross-split source
reuse and more than 15 accepted episodes per source are rejected. Failed recordings
can be marked `failed` and reassigned; they do not count toward accepted coverage.

```bash
python3 "$COBOT_RUN" coverage --index "$COBOT_INDEX" \
  --db "$COBOT_DATA/collection.sqlite" --bundles "$COBOT_DATA/bundles" \
  --out "$COBOT_DATA/coverage.json"
```

Coverage separates tasks, scenes, objects, base-distance bands, clutter and seed
versus expansion slots. Unprepared and blocked work remains visible.

## Verification and limits

The packaged branch passes 15 standard-library tests, validates all 1,000 requests,
and preserves the deployed robot geometry, physical helper classes and controller
settings. The source deployment's local 40-scene sample passed initialization in
all cases; 37 passed conservative approach screening and three need navigation
review. Both repaired cases passed reset, control input, save and exact replay
checks with aligned camera streams.

Fresh physics verification of the packaged branch on RoboVision remains pending
because runtime startup stalled on NFS reads. These checks do not certify task
completion or generalization across the full curriculum. No accepted trajectories,
generated scenes, assignment databases or caches are committed. See
[VERIFICATION.json](../cobot_integration/VERIFICATION.json) for recorded evidence.

```bash
PYTHONPATH=. python3 -m unittest discover -s cobot_scenarios/tests -v
```

The explicit simulator checks in `tests/simulation_smoke.py` exercise reset,
physical inputs, recording and replay. `tests/simulation_start_checks.py` checks
that rejected starts restore the initial state. Their test recordings are not
accepted demonstrations. Automatic MimicGen expansion and VLA training/export are
separate steps to be qualified after seed collection.
