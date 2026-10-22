"""No simulator dependency: planning, immutable IDs, assignment and coverage."""
import ast, hashlib, json, random, sqlite3
from collections import Counter
from pathlib import Path
from . import VERSION
from .spec import TASKS, SCENES, CATEGORIES, VARIATIONS

def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)

def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()

def read(path): return json.loads(Path(path).read_text())
def write_new(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    payload=canonical(value)+'\n'
    try:
        with path.open('x') as f:f.write(payload)
    except FileExistsError:
        if path.read_text()!=payload:raise ValueError(f'Immutable file conflict: {path}')

def inventory(assets):
    root=Path(assets).resolve(); result={}
    # Honor upstream excluded objects without importing the simulator.
    source=root.parent/'objects'/'kitchen_objects.py'
    tree=ast.parse(source.read_text())
    assignment=next(n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='OBJ_CATEGORIES' for t in n.targets))
    definitions={k.arg:k.value for k in assignment.value.keywords}
    for category in CATEGORIES:
        definition={k.arg:k.value for k in definitions[category].keywords}
        paths=[]
        for reg in ('objaverse','lightwheel'):
            if reg not in definition:continue
            options={k.arg:k.value for k in definition[reg].keywords}
            excluded=ast.literal_eval(options['exclude']) if 'exclude' in options else []
            folders=ast.literal_eval(options['model_folders']) if 'model_folders' in options else [f'{reg}/{category}']
            for folder in folders:
                paths.extend(p for p in root.glob(f'objects/{folder}/*/model.xml') if p.is_file() and p.parent.name not in excluded)
        paths=sorted(paths)
        if len(paths)<6:raise ValueError(f'{category}: need six assets, found {len(paths)}')
        result[category]={split:[dict(path=str(p.relative_to(root)),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),qualification='pending') for p in paths[a:b]] for split,a,b in [('train',0,3),('validation',3,4),('test',4,6)]}
    return dict(schema_version=VERSION,objects=result,scenes=SCENES,variations=VARIATIONS,tasks=TASKS)

def generate(registry):
    """Fixed coverage schedule; 96 seed slots are INCLUDED in the 1000 slots."""
    result=[]
    for task,spec in TASKS.items():
        for split,total,seeds in [('train',spec['counts'][0],10),('validation',spec['counts'][1],2)]:
            scene=SCENES[split]
            for i in range(total):
                seed=int(digest([VERSION,task,split,i])[:8],16);rng=random.Random(seed)
                category=spec['objects'][i%len(spec['objects'])] if spec['objects'] else None
                assets=registry['objects'][category][split] if category else []
                asset=assets[(i//max(1,len(spec['objects'])))%len(assets)] if assets else None
                band=[0,1,1,0,1,2,1,0,2,1][i%10]
                clutter=[0,1,2,0,3,1,4,0,2,5][i%10]
                if task=='selective_delivery':clutter=max(1,clutter)
                scene_index=(i//len(spec['envs'])) if len(spec['envs'])>1 else i
                value=dict(schema_version=VERSION,registry_digest=digest(registry),task=task,split=split,index=i,seed=seed,
                    env=spec['envs'][i%len(spec['envs'])],layout=scene['layouts'][scene_index%len(scene['layouts'])],
                    style=scene['styles'][(scene_index//len(scene['layouts']))%len(scene['styles'])],
                    object_category=category,object_asset=asset,
                    seed_slot=f'{task}-{split}-seed-{i%seeds:02d}',collection_kind='human_seed' if i<seeds else 'expansion_candidate',
                    stages=list(spec['stages']),variation=dict(base_band=band,base_distance_m=rng.uniform(*VARIATIONS['base_distance_m'][band]),base_heading_deg=rng.uniform(-90,90),clutter=clutter,object_yaw_deg=rng.uniform(0,360),region_x=rng.uniform(-.8,.8),lighting_scale=rng.uniform(.8,1.2)),
                    qualification='pending')
                pool=[a['path'] for c in CATEGORIES if c!=category for a in registry['objects'][c][split]]
                value['distractor_assets']=[pool[(i+j)%len(pool)] for j in range(clutter)] if category else []
                value['same_category_distractor']=task=='selective_delivery' and i%3==0
                if value['same_category_distractor']:
                    value['distractor_assets'][0]=assets[(i//len(spec['objects'])+1)%len(assets)]['path']
                value['variation']['initial_open_fraction']=(.6+.4*value['variation']['object_yaw_deg']/360 if 'Close' in value['env'] else .1*value['variation']['object_yaw_deg']/360) if task.endswith('_articulation') else 0.
                value['scenario_id']=digest(value)[:20];result.append(value)
    validate(result,registry)
    return result

def validate(manifests,registry):
    ids=set(); asset_splits={}
    expected={(task,split):spec['counts'][i] for task,spec in TASKS.items() for i,split in enumerate(('train','validation'))}
    if Counter((m['task'],m['split']) for m in manifests)!=Counter(expected):raise ValueError('Incomplete or incorrect curriculum quotas')
    for category,partitions in registry['objects'].items():
        for split,assets in partitions.items():
            for asset in assets:
                key=asset['path']
                if key in asset_splits and asset_splits[key]!=split:raise ValueError('Object split leakage')
                asset_splits[key]=split
    for m in manifests:
        body={k:v for k,v in m.items() if k!='scenario_id'}
        if digest(body)[:20]!=m['scenario_id'] or m['scenario_id'] in ids:raise ValueError('Invalid/duplicate scenario ID')
        ids.add(m['scenario_id']);scene=SCENES[m['split']]
        if m['env'] not in TASKS[m['task']]['envs']:raise ValueError('Task/environment mismatch')
        if m['object_asset'] and m['object_asset'] not in registry['objects'][m['object_category']][m['split']]:raise ValueError('Object/category mismatch')
        if m['layout'] not in scene['layouts'] or m['style'] not in scene['styles']:raise ValueError('Scene split leakage')
        if m['registry_digest']!=digest(registry):raise ValueError('Registry mismatch')
        if m['object_asset'] and asset_splits.get(m['object_asset']['path'])!=m['split']:raise ValueError('Asset split mismatch')
    return True

class Ledger:
    """SQLite transactions protect concurrent claims. No implicit acceptance."""
    def __init__(self,path):
        self.db=sqlite3.connect(path,timeout=30)
        self.db.execute('CREATE TABLE IF NOT EXISTS episodes (id TEXT PRIMARY KEY, split TEXT, task TEXT, owner TEXT, status TEXT, revision INTEGER, recording TEXT, lineage TEXT, evidence TEXT)')
        self.db.execute('CREATE TABLE IF NOT EXISTS events (id TEXT, revision INTEGER, payload TEXT, PRIMARY KEY(id,revision))')
    def initialize(self,manifests):
        with self.db:
            for m in manifests:self.db.execute('INSERT OR IGNORE INTO episodes VALUES(?,?,?,NULL,?,0,NULL,NULL,NULL)',(m['scenario_id'],m['split'],m['task'],'available'))
    def update(self,id,owner,revision,status,recording=None,lineage=None,evidence=None):
        if not owner:raise ValueError('Collector name required')
        if status not in ('assigned','recorded','accepted','failed'):raise ValueError('Invalid status')
        with self.db:
            self.db.execute('BEGIN IMMEDIATE')
            row=self.db.execute('SELECT * FROM episodes WHERE id=?',(id,)).fetchone()
            if not row or row[5]!=revision or row[3] not in (None,owner):raise ValueError('Assignment/revision conflict')
            allowed={'available':{'assigned'},'assigned':{'recorded','failed'},'recorded':{'accepted','failed'},'failed':{'assigned'},'accepted':set()}
            if status not in allowed[row[4]]:raise ValueError('Invalid status transition')
            if status=='recorded' and not recording:raise ValueError('Recording path required')
            if status=='accepted':
                if not lineage or not evidence or not all(evidence.get(k) is True for k in ('physical_success','replay','sensor_alignment','collision_review','instruction_review')):raise ValueError('Acceptance evidence required')
                if evidence.get('scenario_id')!=id:raise ValueError('Evidence scenario mismatch')
                others=self.db.execute("SELECT split FROM episodes WHERE lineage=? AND status='accepted'",(lineage,)).fetchall()
                if any(x[0]!=row[1] for x in others):raise ValueError('Lineage split leakage')
                if len(others)>=15:raise ValueError('Lineage cap reached')
            payload=dict(id=id,owner=owner,revision=revision,status=status,recording=recording,lineage=lineage,evidence=evidence)
            self.db.execute('UPDATE episodes SET owner=?,status=?,revision=revision+1,recording=COALESCE(?,recording),lineage=COALESCE(?,lineage),evidence=? WHERE id=?',(owner,status,recording,lineage,canonical(evidence),id))
            self.db.execute('INSERT INTO events VALUES(?,?,?)',(id,revision+1,canonical(payload)))
    def report(self):
        return [dict(split=r[0],task=r[1],status=r[2],count=r[3]) for r in self.db.execute('SELECT split,task,status,COUNT(*) FROM episodes GROUP BY split,task,status')]

def coverage(manifests,ledger,bundles=None):
    states={r[0]:r[1] for r in ledger.db.execute('SELECT id,status FROM episodes')}
    axes={k:Counter() for k in ('task','scene','object_category','base_band','clutter','collection_kind')}
    qualifications=Counter()
    for m in manifests:
        status=states.get(m['scenario_id'],'unassigned')
        values=dict(task=m['task'],scene=f"layout{m['layout']}/style{m['style']}",object_category=m['object_category'] or 'none',base_band=str(m['variation']['base_band']),clutter=str(m['variation']['clutter']),collection_kind=m['collection_kind'])
        for key,value in values.items():axes[key][(m['split'],value,status)]+=1
        if bundles:
            folder=Path(bundles)/m['scenario_id']
            if (folder/'resolved.json').exists():qualifications[read(folder/'resolved.json')['qualification']['initialization']]+=1
            elif (folder/'failure.json').exists():qualifications['blocked']+=1
            else:qualifications['not_prepared']+=1
    return dict(total=len(manifests),accepted=sum(states.get(m['scenario_id'])=='accepted' for m in manifests),remaining=sum(states.get(m['scenario_id'])!='accepted' for m in manifests),initialization=dict(qualifications),axes={k:[dict(split=s,value=v,status=t,count=n) for (s,v,t),n in sorted(counts.items())] for k,counts in axes.items()})
