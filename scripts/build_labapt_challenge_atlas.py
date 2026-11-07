"""Render a separate challenge atlas from the versioned RoboCasa recipe catalog."""
import argparse
import html
import importlib.util
import json
from pathlib import Path


def build(root, output, evidence=None):
    path=Path(root)/'cobot_integration/challenge/catalog.py'
    spec=importlib.util.spec_from_file_location('challenge_catalog',path)
    catalog=importlib.util.module_from_spec(spec);spec.loader.exec_module(catalog)
    checks = json.loads(Path(evidence).read_text()) if evidence else {}
    observed = {r['task']:r for r in checks.get('tasks', [])}
    cards=[]; records=[]
    for name, profile in catalog.PROFILES.items():
        variants=[catalog.recipe(name,d) for d in catalog.DIFFICULTIES]
        r=variants[0]; observation=observed.get(name, {})
        records.append(dict(profile=name,variants=variants,validation=observation))
        illustration = Path(root)/'cobot_integration/challenge/images'/f'{name}.scene.jpg'
        picture = f'<figure><img loading="lazy" src="../cobot_integration/challenge/images/{name}.scene.jpg" alt="Development reset for {name}"><figcaption>Development reset · starting scene only</figcaption></figure>' if illustration.is_file() else ''
        summary = observation.get('status', 'not_run')
        flags = observation.get('result', {}).get('checks', {})
        passed = sum(v is True for k,v in flags.items() if k not in ('physical_fit_review','witness'))
        total = sum(k not in ('physical_fit_review','witness') for k in flags)
        validation = f'Runtime: {summary}; automated checks {passed}/{total}. Geometry review and witness pending.' if observation else 'Recipe implemented. No scored scene or policy success implied.'
        values=' → '.join(str(v['parameters'][r['primary_axis']]) for v in variants)
        cards.append(f'''<article data-family="{r['family']}"><small>{r['family'].upper()} · CUSTOM TASK · EXPERIMENTAL</small>
<h2>{html.escape(name)}</h2>{picture}<p>{html.escape(r['instruction'])}</p>
<dl><dt>Primary variation · easy → medium → hard</dt><dd>{html.escape(r['primary_axis'])}: {values}</dd>
<dt>Initialization</dt><dd>Full workflow / manipulation checkpoint</dd>
<dt>Evidence</dt><dd>{html.escape(validation)}</dd></dl></article>''')
    page='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>LabApt manipulation challenge atlas</title><style>body{font:16px/1.6 system-ui;background:#fafbf7;color:#213930;max-width:1150px;margin:auto;padding:32px}a{color:inherit}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:20px}article{background:white;border:1px solid #dde4d9;border-radius:16px;padding:24px}h2{font-size:21px;overflow-wrap:anywhere}small,dt{color:#68766c}figure{margin:0}img{width:100%;border-radius:8px}figcaption{font-size:12px;color:#68766c}dd{margin:4px 0 18px}button{padding:10px;margin:0 8px 20px 0}aside{border-left:4px solid #71945a;padding:12px 20px;background:#eef2e9}</style>
<a href="../cobot_integration/challenge/README.md">← Task catalog and randomization guide</a><h1>LabApt manipulation challenge</h1>
<p>Ten task profiles · two overlapping families · three difficulty levels · measured layout 100.</p>
<aside>Every profile starts experimental. A scored scene requires verified geometry, cameras, stable reset, complete state replay and a successful witness replay. Policy results and physical validation are separate evidence.</aside>
<h2>What changes?</h2><p>Kitchen geometry stays fixed. Task objects, constrained placements and recorded initialization checkpoints vary. Numerical ranges are development defaults and must pass geometry review before formal scene freezing.</p>
<button onclick="filter('all')">All ten</button><button onclick="filter('constrained')">Constrained manipulation</button><button onclick="filter('contact')">Contact-rich manipulation</button>
<section class="grid">'''+''.join(cards)+'''</section><h2>Evaluation contract</h2><p>Prepare 20 practice and 50 formal scenes per task, difficulty and initialization mode. Methods share frozen scenes, observations, action limits and success rules. Record controller development, demonstration and pretraining budgets separately. Report diagnostic runs separately from qualified formal runs.</p>
<p>Tray guides are removable tabletop apparatus. Wiping scores physical contact coverage, not cleaning chemistry. Rack slot bounds and bowl seated-height intervals require asset-specific collision review. No synthetic force sensor or fixture geometry change is introduced.</p>
<script>function filter(f){document.querySelectorAll('article').forEach(e=>e.hidden=f!=='all'&&e.dataset.family!==f)}</script></html>'''
    output=Path(output);output.write_text(page)
    output.with_suffix('.json').write_text(json.dumps(dict(version=catalog.VERSION,robocasa_commit=checks.get('robocasa_commit'),profiles=records),indent=2)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--robocasa-root',type=Path,required=True)
    p.add_argument('--evidence',type=Path)
    p.add_argument('--output',type=Path,default=Path(__file__).resolve().parents[1]/'docs/LABAPT_CHALLENGE_ATLAS.html')
    a=p.parse_args();build(a.robocasa_root,a.output,a.evidence)


if __name__=='__main__':main()
