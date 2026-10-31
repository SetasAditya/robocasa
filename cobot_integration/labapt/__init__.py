"""Optional development tools for the registered LabApt kitchen.

Normal RoboCasa environments do not import this package.
"""
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
INTEGRATION = ROOT / "cobot_integration"
# Existing shared Cobot helpers use script-style imports.
if str(INTEGRATION) not in sys.path:
    sys.path.insert(0, str(INTEGRATION))


def snapshot_sources(output):
    """Keep the actual controller sources beside each experimental episode."""
    paths = list(Path(__file__).parent.glob("*.py"))
    paths += [INTEGRATION / name for name in (
        "kitchen_tasks.py", "moma_validation.py", "cobot.py", "cobot_fourbar.py",
        "kitchen_recording.py",
    )]
    for source in paths:
        target = output / "source" / source.relative_to(INTEGRATION)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def object_snapshot(env):
    """Measure actual base/object motion and bilateral physical grasp contacts."""
    robot = env.robots[0]
    gripper = robot.gripper["right"]
    groups = [gripper.important_geoms[key] for key in ("left_finger", "right_finger")]
    name = "obj" if "obj" in env.objects else next(iter(env.objects))
    held = False
    for candidate, obj in env.objects.items():
        if env._check_grasp(groups, obj):
            name, held = candidate, True
            break
    return dict(
        base_xy=env.sim.data.get_body_xpos("mobilebase0_base")[:2].copy(),
        object_xyz=env.sim.data.body_xpos[env.obj_body_id[name]].copy(),
        grasped=held, object_name=name,
    )
