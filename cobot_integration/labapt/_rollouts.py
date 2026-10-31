"""Bounded physical/replay/contact checks of five LabApt controller workflows."""
import argparse, concurrent.futures, hashlib, json, os, subprocess, sys
from pathlib import Path
from . import ROOT
TASKS = ('OpenCabinet', 'OpenDrawer', 'OpenFridge', 'OpenMicrowave', 'fetch_surface')

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--seeds', type=int, default=3)
    p.add_argument('--jobs', type=int, default=4)
    args = p.parse_args()
    if args.seeds < 1 or args.jobs < 1:
        p.error('seeds and jobs must be positive')
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    env = os.environ.copy()
    env.update(MUJOCO_GL='egl', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    sources = list(Path(__file__).parent.glob('*.py')) + [ROOT / 'robocasa/models/scenes/labapt.py'] + list((ROOT / 'robocasa/models/assets/fixtures/labapt').glob('*/model.xml'))
    (args.output / 'source_sha256.json').write_text(json.dumps({str(f.relative_to(ROOT)): hashlib.sha256(f.read_bytes()).hexdigest() for f in sources}, indent=2) + '\n')

    def run(task, seed):
        target = args.output / task / f'seed_{seed}'
        module = '._fetch_probe' if task == 'fetch_surface' else '._fixture_probe'
        command = [sys.executable, '-m', 'cobot_integration.labapt' + module, '--seed', str(seed), '--output', str(target)]
        if task != 'fetch_surface':
            command += ['--task', task]
        with (args.output / f'{task}_{seed}.log').open('w') as log:
            try:
                subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=420, check=True)
            except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as exc:
                return dict(task=task, seed=seed, passed=False, error=str(exc))
            result = json.loads((target / 'result.json').read_text())
            dataset = target / 'probe.hdf5'
            if dataset.exists():
                subprocess.run([sys.executable, '-m', 'cobot_integration.labapt._contacts', str(dataset), '--stride', '1'], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=120)
        contact_path = target / 'contact_audit.json'
        contacts = json.loads(contact_path.read_text()) if contact_path.exists() else None
        passed = bool(result.get('success') and result.get('replay_passed', result.get('accepted')) and contacts and (not contacts['penetrations_over_5mm']))
        return dict(task=task, seed=seed, passed=passed, result=result, contacts=contacts, training_eligible=False)
    records = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = [pool.submit(run, task, seed) for seed in range(args.seeds) for task in TASKS]
        for f in concurrent.futures.as_completed(futures):
            row = f.result()
            records.append(row)
            print(json.dumps({k: row[k] for k in ('task', 'seed', 'passed')}), flush=True)
            (args.output / 'summary.json').write_text(json.dumps(dict(results=records, passed=sum((r['passed'] for r in records)), attempts=len(records), training_eligible=False), indent=2) + '\n')
    return 0 if all((r['passed'] for r in records)) else 1
if __name__ == '__main__':
    raise SystemExit(main())
