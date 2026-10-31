"""Inventory installed RoboCasa tasks against LabApt's physical fixture types.

Static candidates are not certificates. Resets and physical rollouts are separate
columns and are populated only from validation evidence.
"""
import ast, inspect, json, argparse, textwrap
from pathlib import Path
import numpy as np
import robocasa
from robosuite.environments.base import REGISTERED_ENVS
from robocasa.environments.kitchen.kitchen import Kitchen
from robocasa.models.scenes.kitchen_arena import KitchenArena
from robocasa.models.fixtures.fixture import FixtureType
from robocasa.models.fixtures.fixture_utils import fixture_is_type

def audit(output, reports):
    arena = KitchenArena(100, 1, rng=np.random.default_rng(0), enable_fixtures=['dish_rack'])
    available = {}
    for t in FixtureType:
        try:
            available[t.name] = [name for name, f in arena.fixtures.items() if fixture_is_type(f, t)]
        except ValueError:
            available[t.name] = []
    evidence = {}
    for path in reports:
        for item in json.loads(path.read_text()):
            evidence.setdefault(item['task'], []).append(item)
    records = []
    for name, cls in sorted(REGISTERED_ENVS.items()):
        if not issubclass(cls, Kitchen) or cls is Kitchen:
            continue
        if not cls.__module__.startswith('robocasa.environments.kitchen'):
            continue
        required = set()
        dynamic = []
        affordances = set()
        for base in cls.__mro__:
            if base is Kitchen:
                break
            for method_name in ('__init__', '_setup_kitchen_references'):
                method = base.__dict__.get(method_name)
                if method is None:
                    continue
                try:
                    tree = ast.parse(textwrap.dedent(inspect.getsource(method)))
                except (TypeError, OSError):
                    continue
                for node in ast.walk(tree):
                    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and (node.func.attr == 'get_fixture') and node.args:
                        v = node.args[0]
                        if isinstance(v, ast.Attribute) and isinstance(v.value, ast.Name) and (v.value.id == 'FixtureType'):
                            required.add(v.attr)
                    if isinstance(node, ast.keyword) and node.arg in ('id', 'fixture_id', 'drawer_id', 'cab_id'):
                        v = node.value
                        if isinstance(v, ast.Attribute) and isinstance(v.value, ast.Name) and (v.value.id == 'FixtureType'):
                            required.add(v.attr)
                    if isinstance(node, ast.arguments):
                        for arg, default in zip(node.args[-len(node.defaults):], node.defaults) if node.defaults else []:
                            if arg.arg.endswith('_id') and isinstance(default, ast.Attribute) and isinstance(default.value, ast.Name) and (default.value.id == 'FixtureType'):
                                required.add(default.attr)
        try:
            params = inspect.signature(cls).parameters
            required_args = [k for k, v in params.items() if v.default is inspect.Parameter.empty and v.kind not in (v.VAR_POSITIONAL, v.VAR_KEYWORD)]
        except (ValueError, TypeError):
            required_args = []
        for base in cls.__mro__:
            if base is Kitchen:
                break
            try:
                source = inspect.getsource(base)
            except (TypeError, OSError):
                continue
            if 'self.fridge' in source and ('reg_type="drawer"' in source or 'reg_type="freezer_drawer"' in source):
                affordances.add('refrigerator_drawer')
        missing = sorted((t for t in required if not available.get(t)))
        tests = evidence.get(name, [])
        status = 'blocked_missing_fixture' if missing else 'candidate_requires_runtime_validation'
        if affordances:
            status = 'blocked_missing_affordance'
        if required_args:
            status = 'requires_task_parameters'
        if tests:
            status = 'reset_pass' if all((t['status'] == 'reset_pass' for t in tests)) else 'reset_fail'
            if status == 'reset_pass' and any((t.get('initial_penetrations_over_5mm') for t in tests)):
                status = 'reset_contact_failure'
        records.append(dict(task=name, module=cls.__module__, required_fixture_types=sorted(required), missing_affordances=sorted(affordances), required_parameters=required_args, missing_fixture_types=missing, status=status, reset_passes=sum((t['status'] == 'reset_pass' for t in tests)), reset_attempts=len(tests), contact_clean_resets=sum((t['status'] == 'reset_pass' and (not t.get('initial_penetrations_over_5mm')) for t in tests)), physical_rollout='unvalidated', limitations='Static dependency extraction is conservative and incomplete for dynamic fixture selection; exact spatial and asset requirements need runtime checks.'))
    result = dict(layout=100, available_fixture_types=available, tasks=records, summary={s: sum((r['status'] == s for r in records)) for s in sorted({r['status'] for r in records})}, notes=['No oven, stove, coffee machine, blender, toaster, kettle, or stand mixer has been added to fabricate task compatibility.', 'Fridge drawers are absent from the fitted two-door source asset; tasks requiring them are not supported.', 'Existing asset catalog and original task success checks are unchanged.'])
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result['summary']))
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--reports', nargs='*', type=Path, default=[])
    a = p.parse_args()
    audit(a.output, a.reports)
