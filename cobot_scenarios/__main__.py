import argparse,json,sys,traceback
from pathlib import Path
from .core import inventory,generate,read,write_new,Ledger,validate,coverage

def main():
    p=argparse.ArgumentParser(description='CoBot shared scenario suite')
    p.add_argument('--integration',help='RoboCasa cobot_integration directory (simulation only)')
    sub=p.add_subparsers(dest='cmd',required=True)
    s=sub.add_parser('inventory');s.add_argument('--assets',required=True);s.add_argument('--out',required=True)
    s=sub.add_parser('generate');s.add_argument('--registry',required=True);s.add_argument('--out',required=True);s.add_argument('--index-only',action='store_true')
    s=sub.add_parser('check');s.add_argument('--registry',required=True);g=s.add_mutually_exclusive_group(required=True);g.add_argument('--scenarios');g.add_argument('--index')
    s=sub.add_parser('gallery');s.add_argument('--bundles',action='append',required=True);s.add_argument('--out',required=True)
    s=sub.add_parser('coverage');s.add_argument('--index',required=True);s.add_argument('--db',required=True);s.add_argument('--bundles');s.add_argument('--out',required=True)
    s=sub.add_parser('prepare');s.add_argument('--manifest');s.add_argument('--index');s.add_argument('--id');s.add_argument('--assets',required=True);s.add_argument('--out',required=True);s.add_argument('--no-render',action='store_true')
    s=sub.add_parser('prepare-batch');s.add_argument('--index',required=True);s.add_argument('--assets',required=True);s.add_argument('--out',required=True);s.add_argument('--limit',type=int,default=8);s.add_argument('--no-render',action='store_true')
    s=sub.add_parser('restore-check');s.add_argument('--bundle',required=True)
    s=sub.add_parser('serve');s.add_argument('--bundle',required=True);s.add_argument('--recordings',required=True);s.add_argument('--port',type=int,default=8766)
    s=sub.add_parser('replay');s.add_argument('--bundle',required=True);s.add_argument('--recording',required=True)
    s=sub.add_parser('ledger');s.add_argument('--db',required=True);s.add_argument('--scenarios');s.add_argument('--index');s.add_argument('--id');s.add_argument('--owner');s.add_argument('--revision',type=int,default=0);s.add_argument('--status',choices=['assigned','recorded','accepted','failed']);s.add_argument('--recording');s.add_argument('--lineage');s.add_argument('--evidence');s.add_argument('--merge');s.add_argument('--export')
    a=p.parse_args()
    if a.cmd=='inventory':write_new(a.out,inventory(a.assets));print(a.out)
    elif a.cmd=='generate':
        rows=generate(read(a.registry))
        if not a.index_only:
            for row in rows:write_new(Path(a.out)/(row['scenario_id']+'.json'),row)
        write_new(Path(a.out).parent/'scenario_index.json',rows)
        print(json.dumps(dict(scenarios=len(rows),human_seeds=sum(r['collection_kind']=='human_seed' for r in rows))))
    elif a.cmd=='check':
        rows=read(a.index) if a.index else [read(x) for x in sorted(Path(a.scenarios).glob('*.json'))];validate(rows,read(a.registry));print(json.dumps(dict(valid=True,count=len(rows))))
    elif a.cmd=='gallery':
        from .reporting import gallery
        print(json.dumps(gallery(a.bundles,a.out)))
    elif a.cmd=='coverage':
        ledger=Ledger(a.db)
        result=coverage(read(a.index),ledger,a.bundles);ledger.db.close()
        Path(a.out).write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='axes'}))
    elif a.cmd=='ledger':
        ledger=Ledger(a.db)
        if a.index:ledger.initialize(read(a.index))
        if a.scenarios:ledger.initialize([read(x) for x in Path(a.scenarios).glob('*.json')])
        if a.merge:
            for event in read(a.merge):
                existing=ledger.db.execute('SELECT payload FROM events WHERE id=? AND revision=?',(event['id'],event['revision']+1)).fetchone()
                if existing:
                    if json.loads(existing[0])!=event:raise ValueError('Conflicting imported event')
                else:ledger.update(**event)
        if a.id:ledger.update(a.id,a.owner,a.revision,a.status,a.recording,a.lineage,read(a.evidence) if a.evidence else None)
        if a.export:write_new(a.export,[json.loads(r[0]) for r in ledger.db.execute('SELECT payload FROM events ORDER BY rowid')])
        print(json.dumps(ledger.report(),indent=2));ledger.db.close()
    else:
        if not a.integration:p.error('--integration required for simulation commands')
        from . import runtime
        runtime.bootstrap(a.integration)
        if a.cmd=='prepare':
            try:
                m=read(a.manifest) if a.manifest else next(r for r in read(a.index) if r['scenario_id']==a.id)
                r=runtime.prepare(m,a.assets,a.out,not a.no_render);print(json.dumps(r['qualification']))
            except Exception as e:
                Path(a.out).mkdir(parents=True,exist_ok=True)
                write_new(Path(a.out)/'failure.json',dict(status='blocked',error=repr(e),traceback=traceback.format_exc()))
                raise
        elif a.cmd=='prepare-batch':
            rows=read(a.index)
            rows=sorted(rows,key=lambda r:(r['index'],r['split'],r['task']))[:a.limit]
            report=[]
            for row in rows:
                out=Path(a.out)/row['scenario_id']
                try:
                    result=read(out/'resolved.json') if (out/'resolved.json').exists() else runtime.prepare(row,a.assets,out,not a.no_render)
                    entry=dict(scenario_id=row['scenario_id'],task=row['task'],qualification=result['qualification'])
                except Exception as e:
                    entry=dict(scenario_id=row['scenario_id'],task=row['task'],status='blocked',error=repr(e))
                    out.mkdir(parents=True,exist_ok=True)
                    (out/'failure.json').write_text(json.dumps(entry,indent=2))
                report.append(entry);print(json.dumps(entry),flush=True)
            (Path(a.out)/'batch_report.json').write_text(json.dumps(report,indent=2))
        elif a.cmd=='restore-check':
            env,r=runtime.restore(a.bundle,False);env.close();print(json.dumps(dict(exact_initial_state=True,scenario_id=r['scenario_id'])))
        elif a.cmd=='serve':
            from .browser import serve
            serve(a.bundle,a.recordings,a.port)
        elif a.cmd=='replay':
            from .browser import replay
            print(json.dumps(replay(a.bundle,a.recording),indent=2))
if __name__=='__main__':main()
