"""Dataset plans and immutable episode lineage. No simulator dependencies."""
import hashlib
from itertools import product
import json
from pathlib import Path

VERSION = 'cobot-robocasa-data-v1'
SPLITS = ('train', 'validation', 'test')
LAYOUTS = {'train': (11,16,24,38), 'validation': (45,), 'test': (100,)}
STYLES = {'train': (11,22), 'validation': (33,44), 'test': (1,)}
NATIVE_TASKS = ('OpenDrawer','CloseDrawer','OpenCabinet','CloseCabinet','OpenMicrowave','CloseMicrowave')


def identity(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def file_sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value, *, overwrite=False):
    path=Path(path)
    with path.open('w' if overwrite else 'x') as f:
        json.dump(value,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')


def kitchen_policy(config=None):
    policy=config or {'layouts':LAYOUTS,'styles':STYLES}
    if set(policy)!={'layouts','styles'}:raise ValueError('Kitchen config needs layouts and styles')
    out={kind:{split:list(policy[kind][split]) for split in SPLITS} for kind in ('layouts','styles')}
    for kind in out:
        for values in out[kind].values():
            if not values or any(type(v) is not int or v<1 or v>100 for v in values):raise ValueError('Use explicit positive layout/style IDs')
    a,b,c=(set(out['layouts'][s]) for s in SPLITS)
    if a&b or a&c or b&c or 100 in a|b or c!={100}:
        raise ValueError('Transfer kitchens must be disjoint, with LabApt reserved for test')
    if out['styles']['test']!=[1]:raise ValueError('Measured LabApt test appearance is style 1')
    return out


def check_request(r):
    if r['track'] not in ('transfer','challenge') or r['split'] not in SPLITS:
        raise ValueError('Unknown track or split')
    if r['track']=='transfer':
        if not r['task'].isidentifier() or r['task'].startswith('LabApt'):
            raise ValueError('Transfer requires a native RoboCasa environment name')
        policy=kitchen_policy(r.get('kitchen_config'))
        if r['layout'] not in policy['layouts'][r['split']] or r['style'] not in policy['styles'][r['split']]:
            raise ValueError('Layout/style violates transfer kitchen holdouts')
    else:
        from ..challenge.catalog import recipe
        recipe(r['task'],r['difficulty'],r['mode'])
        if r['layout']!=100 or r['style']!=1:raise ValueError('Challenge requires measured layout 100/style 1')
    if r['asset_split'] != ('pretrain' if r['split']=='train' else 'target'):
        raise ValueError('Object split does not match dataset split')
    if r['seed']!=int(identity({k:v for k,v in r.items() if k not in ('id','lineage','seed')})[:8],16) % (2**31):
        raise ValueError('Scene seed does not match its planned split')
    expected=identity({k:v for k,v in r.items() if k not in ('id','lineage')})
    if r['id']!=expected[:24]:raise ValueError('Request identity changed')
    if r['lineage']!=identity({k:v for k,v in r.items() if k not in ('id','lineage','split','asset_split')}):
        raise ValueError('Invalid scene lineage')


def plan(track, tasks=None, count=10, difficulties=('easy',), modes=('workflow',),kitchen_config=None):
    if type(count) is not int or count<1:raise ValueError('count must be positive')
    if track not in ('transfer','challenge'):raise ValueError('Unknown track')
    if tasks is None:
        if track=='transfer':tasks=NATIVE_TASKS
        else:
            from ..challenge.catalog import PROFILES
            tasks=tuple(PROFILES)
    if track=='challenge' and kitchen_config is not None:raise ValueError('Challenge kitchen geometry is fixed')
    policy=kitchen_policy(kitchen_config)
    rows=[]
    for split in SPLITS:
        pairs=list(product(policy['layouts'][split],policy['styles'][split]))
        for task in tasks:
            for difficulty in difficulties if track=='challenge' else ('native',):
                for mode in modes if track=='challenge' else ('workflow',):
                    for index in range(count):
                        r=dict(version=VERSION,track=track,split=split,task=task,difficulty=difficulty,mode=mode,
                               layout=100 if track=='challenge' else pairs[index%len(pairs)][0],
                               style=1 if track=='challenge' else pairs[index%len(pairs)][1],
                               asset_split='pretrain' if split=='train' else 'target',index=index)
                        if kitchen_config is not None:r['kitchen_config']=policy
                        r['seed']=int(identity(r)[:8],16) % (2**31)
                        r['lineage']=identity({k:v for k,v in r.items() if k not in ('split','asset_split')})
                        r['id']=identity({k:v for k,v in r.items() if k!='lineage'})[:24]
                        check_request(r);rows.append(r)
    validate_plan(dict(version=VERSION,requests=rows))
    return dict(version=VERSION,requests=rows)


def validate_plan(p):
    if p['version']!=VERSION:raise ValueError('Unsupported data version')
    ids=set();seeds=set();lineages={};policies=set()
    for r in p['requests']:
        check_request(r)
        if r['track']=='transfer':policies.add(identity(kitchen_policy(r.get('kitchen_config'))))
        if len(policies)>1:raise ValueError('Cannot combine inconsistent kitchen split policies')
        if r['id'] in ids or r['seed'] in seeds:raise ValueError('Duplicate scene or seed')
        ids.add(r['id']);seeds.add(r['seed'])
        previous=lineages.setdefault(r['lineage'],r['split'])
        if previous!=r['split']:raise ValueError('Lineage leaks across splits')
    return p
