"""Native RoboCasa teleop → auditable demonstration → LeRobot → π0.5."""
import argparse
import json
from pathlib import Path
import sys
from .manifest import plan,read,write,validate_plan


def positive(value):
    number=int(value)
    if number<1:raise argparse.ArgumentTypeError('must be positive')
    return number


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    q=sub.add_parser('plan');q.add_argument('--track',choices=('transfer','challenge'),required=True)
    q.add_argument('--kitchen-config',type=Path);q.add_argument('--tasks',nargs='+');q.add_argument('--count',type=positive,default=10)
    q.add_argument('--difficulties',nargs='+',default=['easy']);q.add_argument('--modes',nargs='+',default=['workflow'])
    q.add_argument('--output',type=Path,required=True)
    q=sub.add_parser('prepare');q.add_argument('--plan',type=Path,required=True);q.add_argument('--id')
    q.add_argument('--index',type=int,default=0);q.add_argument('--scene',type=Path,required=True)
    q.add_argument('--geometry',type=Path);q.add_argument('--initialization',type=Path)
    for name in ('teleop','collect'):
        q=sub.add_parser(name);q.add_argument('--scene',type=Path,required=True);q.add_argument('--output',type=Path,required=True)
        q.add_argument('--max-steps',type=positive,default=1800)
        if name=='teleop':q.add_argument('--device',choices=('keyboard','spacemouse'),default='keyboard')
        else:q.add_argument('--expert',required=True);q.add_argument('--privileged',action='store_true')
    q=sub.add_parser('validate');q.add_argument('--episode',type=Path,required=True);q.add_argument('--review',type=Path)
    q=sub.add_parser('export');q.add_argument('--episodes',nargs='+',type=Path,required=True);q.add_argument('--output',type=Path,required=True)
    q.add_argument('--track',choices=('transfer','challenge'),required=True);q.add_argument('--split',choices=('train','validation','test'),required=True)
    q.add_argument('--repo-id',default='local/cobot-robocasa')
    q=sub.add_parser('inspect');q.add_argument('--dataset',type=Path,required=True)
    q=sub.add_parser('train-smoke');q.add_argument('--dataset',type=Path,required=True);q.add_argument('--output',type=Path,required=True)
    q.add_argument('--python',default=sys.executable);q.add_argument('--lerobot-root',type=Path)
    q.add_argument('--model',type=Path);q.add_argument('--tokenizer',type=Path);q.add_argument('--execute',action='store_true')
    q=sub.add_parser('evaluate');q.add_argument('--scene',type=Path,required=True);q.add_argument('--checkpoint',type=Path,required=True)
    q.add_argument('--output',type=Path,required=True);q.add_argument('--python',required=True)
    q.add_argument('--max-steps',type=positive,default=1800);q.add_argument('--witness',type=Path)
    a=p.parse_args()
    if a.command=='plan':
        result=plan(a.track,a.tasks,a.count,a.difficulties,a.modes,kitchen_config=read(a.kitchen_config) if a.kitchen_config else None);write(a.output,result)
        result={'plan':str(a.output),'requests':len(result['requests']),'first_ids':[r['id'] for r in result['requests'][:5]]}
    elif a.command=='prepare':
        from .runtime import prepare
        manifest=validate_plan(read(a.plan))
        rows=[r for r in manifest['requests'] if r['id']==a.id] if a.id else [manifest['requests'][a.index]]
        if len(rows)!=1:raise ValueError('Unknown request ID')
        result=prepare(rows[0],a.scene,geometry=read(a.geometry) if a.geometry else None,initialization=a.initialization)
    elif a.command=='teleop':
        from .recording import teleop
        result=teleop(a.scene,a.output,a.device,a.max_steps)
    elif a.command=='collect':
        from .recording import collect
        result=collect(a.scene,a.output,a.expert,a.max_steps,a.privileged)
    elif a.command=='validate':
        from .validation import validate
        result=validate(a.episode,a.review)
    elif a.command=='export':
        from .export import export
        result=export(a.episodes,a.output,a.track,a.split,a.repo_id)
    elif a.command=='inspect':
        from .export import inspect
        result=inspect(a.dataset)
    elif a.command=='train-smoke':
        from .training import smoke
        result=smoke(a.dataset,a.output,python=a.python,lerobot_root=a.lerobot_root,model=a.model,tokenizer=a.tokenizer,execute=a.execute)
    else:
        from .evaluation import evaluate
        result=evaluate(a.scene,a.checkpoint,a.output,a.python,a.max_steps,a.witness)
    print(json.dumps(result,indent=2))
    if a.command in ('validate','train-smoke') and not result.get('passed'):return 2
    return 0


if __name__=='__main__':raise SystemExit(main())
