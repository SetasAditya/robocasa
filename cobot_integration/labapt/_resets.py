"""Bounded, isolated checks of every fixture-compatible LabApt task.

Each task gets seed 0 first, then seeds 1..9 if seed 0 initialized. Eight worker
processes are used by default; no policy rollouts or success predicates are
modified. Partial progress is checkpointed and completed jobs can be resumed.
"""
import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback
from . import ROOT
CUSTOM = ['fetch_surface', 'bin_placement', 'collect_two_stations']

def atomic_json(path, data):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, indent=2, default=str) + '\n')
    temporary.replace(path)

def failure_category(error, trace):
    text = str(error) + '\n' + trace
    if isinstance(error, TimeoutError):
        return 'initialization_timeout'
    if isinstance(error, FileNotFoundError):
        return 'missing_asset'
    if isinstance(error, TypeError):
        return 'incompatible_api_or_parameters'
    if isinstance(error, NotImplementedError):
        return 'unsupported_task_or_parameters'
    if '50 times' in text or 'PlacementError' in text:
        return 'insufficient_space_or_placement_failure'
    if 'SamplingError' in text:
        return 'incompatible_asset_or_reset_region'
    if 'drawer' in text and 'fridge' in text:
        return 'missing_fridge_affordance'
    if 'get_fixture' in text and ('cannot be empty' in text or 'AssertionError' in text):
        return 'missing_fixture_or_region'
    return 'runtime_failure'

def worker(task, output):
    os.environ.setdefault('MUJOCO_GL', 'egl')
    sys.path[:0] = [str(ROOT), str(ROOT / 'cobot_integration')]
    import kitchen_tasks
    rows = []

    def timeout(*_):
        raise TimeoutError('Initialization exceeded 90 seconds')
    signal.signal(signal.SIGALRM, timeout)
    for seed in range(10):
        env = None
        start = time.monotonic()
        row = dict(task=task, seed=seed, layout=100)
        try:
            signal.alarm(90)
            env = kitchen_tasks.make_env(task, seed, layout=100, style=1, render=False, images=False)
            env.reset()
            env.sim.forward()
            contacts = []
            for contact in env.sim.data.contact:
                if contact.dist < -0.005:
                    contacts.append(dict(geoms=[env.sim.model.geom_id2name(int(g)) or '' for g in (contact.geom1, contact.geom2)], depth_m=float(-contact.dist)))
            try:
                already_successful = bool(env._check_success())
                predicate_error = None
            except Exception as exc:
                already_successful = None
                predicate_error = type(exc).__name__ + ': ' + str(exc)
            signal.alarm(0)
            row.update(status='reset_pass', contact_check='fail' if contacts else 'pass', initial_penetrations_over_5mm=contacts, initial_success=already_successful, success_predicate_error=predicate_error, action_dim=env.action_dim, fixtures={k: (v[0] if isinstance(v, tuple) else v).name for k, v in env.fixture_refs.items()}, base_position=list(env.init_robot_base_pos), objects=list(env.objects), robot_rollout='not tested')
            row['qualified_reset'] = not contacts and already_successful is False and (predicate_error is None)
            if contacts:
                robot_contact = any((any((n.startswith(('robot', 'mobilebase', 'gripper')) for n in c['geoms'])) for c in contacts))
                row['failure_category'] = 'robot_initialization_contact' if robot_contact else 'initial_object_contact'
            elif predicate_error:
                row['failure_category'] = 'unsupported_success_predicate'
            elif already_successful:
                row['failure_category'] = 'already_successful_at_reset'
        except Exception as exc:
            trace = traceback.format_exc()
            row.update(status='reset_fail', qualified_reset=False, error=type(exc).__name__ + ': ' + str(exc), failure_category=failure_category(exc, trace), traceback=trace[:8000] + '\n...\n' + trace[-4000:] if len(trace) > 12000 else trace)
        finally:
            signal.alarm(0)
            if env is not None:
                try:
                    env.close()
                except Exception:
                    pass
        row['seconds'] = time.monotonic() - start
        rows.append(row)
        atomic_json(output / 'resets.json', rows)
        print(json.dumps({k: v for k, v in row.items() if k not in ('traceback', 'initial_penetrations_over_5mm')}, default=str), flush=True)
        if seed == 0 and row['status'] != 'reset_pass':
            break
    atomic_json(output / 'complete.json', dict(task=task, attempts=len(rows)))

def supervise(task, output):
    directory = output / 'tasks' / task
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / 'complete.json').exists():
        return task
    env = os.environ.copy()
    env.update(MUJOCO_GL='egl', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1')
    with (directory / 'worker.log').open('w') as log:
        process = subprocess.Popen([sys.executable, '-m', 'cobot_integration.labapt._resets', '--worker', task, '--output', str(directory)], stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True, cwd=ROOT)
        last = time.monotonic()
        last_size = 0
        watchdog = False
        while process.poll() is None:
            time.sleep(2)
            checkpoint = directory / 'resets.json'
            size = checkpoint.stat().st_size if checkpoint.exists() else 0
            if size != last_size:
                last = time.monotonic()
                last_size = size
            if time.monotonic() - last > 125:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
                watchdog = True
                break
    if not (directory / 'complete.json').exists():
        atomic_json(directory / 'complete.json', dict(task=task, worker_error='watchdog_timeout' if watchdog else f'exit_{process.returncode}'))
    return task

def summary(output, inventory):
    records = []
    all_rows = []
    seeds = {}
    for candidate in inventory['tasks']:
        row = dict(candidate)
        task = row['task']
        directory = output / 'tasks' / task
        path = directory / 'resets.json'
        tests = json.loads(path.read_text()) if path.exists() else []
        if tests:
            all_rows.extend(tests)
            seeds[task] = [r['seed'] for r in tests if r.get('qualified_reset')]
            row.update(reset_attempts=len(tests), reset_passes=sum((r['status'] == 'reset_pass' for r in tests)), qualified_reset_count=len(seeds[task]), tested_seeds=[r['seed'] for r in tests], failure_categories=sorted({r['failure_category'] for r in tests if 'failure_category' in r}), physical_rollout='unvalidated')
            row['status'] = 'reset_validated' if len(tests) == 10 and len(seeds[task]) == 10 else 'reset_exceptions' if seeds[task] else 'runtime_incompatible'
        completion = directory / 'complete.json'
        if completion.exists():
            detail = json.loads(completion.read_text())
            if 'worker_error' in detail:
                row.update(status='worker_failure', worker_error=detail['worker_error'])
        records.append(row)
    counts = {status: sum((r['status'] == status for r in records)) for status in sorted({r['status'] for r in records})}
    result = dict(layout=100, tasks=records, summary=counts, reset_attempts=len(all_rows), reset_passes=sum((r['status'] == 'reset_pass' for r in all_rows)), qualified_resets=sum((r.get('qualified_reset', False) for r in all_rows)), contact_threshold_m=0.005, physical_rollout='not qualified', notes=['Every selected concrete task gets seed 0; seeds 1..9 run only after successful initialization.', 'Qualified resets additionally require a working success predicate, an initially incomplete task and no contact deeper than 5 mm.', 'This report does not establish physical task completion or real-world transfer.'])
    atomic_json(output / 'task_compatibility.json', result)
    atomic_json(output / 'validated_resets.json', all_rows)
    atomic_json(output / 'reset_checked_seeds.json', dict(layout=100, qualification='reset only', seeds=seeds))
    return result

def run(output, jobs, only):
    output.mkdir(parents=True, exist_ok=True)
    inventory_path = output / 'inventory.json'
    if not inventory_path.exists():
        subprocess.run([sys.executable, '-m', 'cobot_integration.labapt._inventory', '--output', str(inventory_path)], check=True, cwd=ROOT)
        inventory = json.loads(inventory_path.read_text())
        inventory['tasks'].extend((dict(task=t, status='candidate_requires_runtime_validation', module='cobot_integration.kitchen_tasks') for t in CUSTOM))
        atomic_json(inventory_path, inventory)
        from robocasa.models.scenes.scene_registry import get_layout_path
        layout = Path(get_layout_path(100))
        inputs = [layout.with_suffix(suffix) for suffix in ('.yaml', '.manifest.json', '.walls.json')]
        atomic_json(output / 'scene_hashes.json', {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs})
    inventory = json.loads(inventory_path.read_text())
    tasks = [r['task'] for r in inventory['tasks'] if r['status'] == 'candidate_requires_runtime_validation']
    if only:
        tasks = [t for t in tasks if t in only]
    with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as pool:
        futures = [pool.submit(supervise, t, output) for t in tasks]
        for index, future in enumerate(concurrent.futures.as_completed(futures), 1):
            task = future.result()
            result = summary(output, inventory)
            print(json.dumps(dict(completed=index, total=len(tasks), task=task, summary=result['summary'])), flush=True)
    result = summary(output, inventory)
    print(json.dumps(dict(done=True, summary=result['summary'], reset_attempts=result['reset_attempts'], qualified_resets=result['qualified_resets'])), flush=True)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--worker')
    parser.add_argument('--jobs', type=int, default=8)
    parser.add_argument('--only', nargs='*')
    args = parser.parse_args()
    if args.worker:
        worker(args.worker, args.output)
    else:
        run(args.output, args.jobs, args.only)
