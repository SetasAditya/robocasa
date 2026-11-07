"""Pinned LeRobot π0.5 recipe, data-loader checks and bounded training launcher."""
import os
from pathlib import Path
import subprocess
import sys
from .manifest import read,write,file_sha

LEROBOT_COMMIT='b9cb121cb7d4e3e26ec5c906d08dda68d151ba52'
MODEL='lerobot/pi05_base'
MODEL_REVISION='22c30e5b34f9e7ab738b1b34fc2fc5ecfe2d28f0'
TOKENIZER='google/paligemma-3b-pt-224'
TOKENIZER_REVISION='35e4f46485b4d07967e7e9935bc3786aad50687c'


def recipe(dataset,output,model,tokenizer,steps=2):
    dataset=Path(dataset);m=read(dataset/'collection.json');check=read(dataset/'inspection.json')
    if m['split']!='train' or m['purpose']!='training':raise ValueError('Training requires the training partition')
    if not check['passed'] or check['collection_sha256']!=file_sha(dataset/'collection.json'):
        raise ValueError('Dataset inspection missing or stale')
    for name,sha in m['files'].items():
        if file_sha(dataset/name)!=sha:raise ValueError('Training data changed')
    return dict(dataset=dict(repo_id=m['repo_id'],root=str(dataset.resolve()),video_backend='pyav'),
        policy=dict(type='pi05',pretrained_path=str(Path(model).resolve()),text_tokenizer_name=str(Path(tokenizer).resolve()),
            device='cuda',dtype='float32',chunk_size=50,n_action_steps=5,num_inference_steps=10,
            gradient_checkpointing=True,compile_model=False,train_expert_only=True,freeze_vision_encoder=True,
            use_relative_actions=False,push_to_hub=False),
        output_dir=str(Path(output).resolve()),job_name='cobot-pi05-smoke',batch_size=1,num_workers=0,
        steps=steps,log_freq=1,save_freq=steps,env_eval_freq=0,eval_steps=0,max_eval_samples=0,seed=7,wandb=dict(enable=False))


def loader_check(dataset):
    import numpy as np
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    dataset=Path(dataset);m=read(dataset/'collection.json')
    if m['split']!='train':raise ValueError('Only training data can provide normalization')
    ds=LeRobotDataset(m['repo_id'],root=dataset,video_backend='pyav',delta_timestamps={'action':[i/20 for i in range(50)]})
    offset=0
    for ep in m['episodes']:
        last=ds[offset+ep['frames']-1]
        if tuple(last['action'].shape)!=(50,m['contract']['action_dim']):raise ValueError('Wrong action chunk dimensions')
        pad=last['action_is_pad'];pad=pad.numpy() if hasattr(pad,'numpy') else np.asarray(pad)
        if int(pad.sum())!=49:raise ValueError('Terminal chunk crosses episode boundary')
        offset+=ep['frames']
    if len(ds)!=m['frames']:raise ValueError('Training reader row count changed')
    import importlib.metadata
    return dict(passed=True,reader_lerobot_version=importlib.metadata.version('lerobot'),reader_python=sys.version.split()[0],frames=len(ds),episodes=len(m['episodes']),action_horizon=50,
                train_only_statistics=True,contract_hash=m['contract']['hash'])


def snapshot_identity(path,repo,revision):
    """Require a pinned HF cache snapshot or an explicit hashed local snapshot record."""
    path=Path(path).resolve()
    if path.parent.name=='snapshots' and path.name==revision:
        return dict(repository=repo,revision=revision,path=str(path))
    manifest=path/'snapshot.json'
    if not manifest.is_file():raise ValueError('Snapshot provenance missing: '+str(path))
    record=read(manifest)
    if record.get('repository')!=repo or record.get('revision')!=revision or not record.get('files'):
        raise ValueError('Wrong pretrained snapshot provenance')
    for name,sha in record['files'].items():
        target=(path/name).resolve()
        if not target.is_relative_to(path) or file_sha(target)!=sha:raise ValueError('Snapshot content changed')
    return record


def smoke(dataset,output,*,python=sys.executable,lerobot_root=None,model=None,tokenizer=None,execute=False):
    import json
    output=Path(output)
    if output.exists():raise FileExistsError(output)
    output.mkdir(parents=True)
    report=dict(data_loader_passed=False,pretrained_backward_passed=False,training_run=False,blocked=[])
    # The reader check runs in the selected dependency interpreter, not the simulator.
    env=dict(os.environ);repo=Path(__file__).resolve().parents[2]
    env['PYTHONPATH']=str(repo)+os.pathsep+env.get('PYTHONPATH','')
    code='import json,sys;from cobot_integration.data.training import loader_check;print("RESULT="+json.dumps(loader_check(sys.argv[1])))'
    with (output/'reader.log').open('w') as log:
        result=subprocess.run([str(python),'-c',code,str(Path(dataset).resolve())],env=env,stdout=log,stderr=subprocess.STDOUT,timeout=300)
    if result.returncode:report['blocked'].append('LeRobot reader check failed; see reader.log')
    else:
        report['data_loader_passed']=True
        lines=[line[7:] for line in (output/'reader.log').read_text().splitlines() if line.startswith('RESULT=')]
        report['reader']=json.loads(lines[-1])
    if not lerobot_root:report['blocked'].append('Pinned LeRobot training checkout not supplied')
    else:
        revision=subprocess.check_output(['git','-C',str(lerobot_root),'rev-parse','HEAD'],text=True).strip()
        if revision!=LEROBOT_COMMIT:report['blocked'].append('Training checkout is not the pinned LeRobot revision')
    for path,label in ((model,'pretrained model'),(tokenizer,'tokenizer')):
        if not path or not Path(path).is_dir():report['blocked'].append(f'Local pinned {label} snapshot not supplied')
        else:
            try:snapshot_identity(path,MODEL if label=='pretrained model' else TOKENIZER,MODEL_REVISION if label=='pretrained model' else TOKENIZER_REVISION)
            except ValueError as exc:report['blocked'].append(str(exc))
    report['pins']=dict(lerobot=LEROBOT_COMMIT,model=MODEL,model_revision=MODEL_REVISION,
                        tokenizer=TOKENIZER,tokenizer_revision=TOKENIZER_REVISION)
    if not report['blocked']:
        config=recipe(dataset,output/'training',model,tokenizer)
        write(output/'train.json',config)
        if execute:
            # Verify the actual imported module, GPU and local model provenance before training.
            preflight='''import pathlib,sys,torch,lerobot
root=pathlib.Path(sys.argv[1]).resolve()
assert pathlib.Path(lerobot.__file__).resolve().is_relative_to(root), 'Imported another LeRobot checkout'
assert torch.cuda.is_available(), 'CUDA unavailable'
print(torch.cuda.get_device_name())
'''
            with (output/'gpu-preflight.log').open('w') as log:
                check=subprocess.run([str(python),'-c',preflight,str(lerobot_root)],env=env,stdout=log,stderr=subprocess.STDOUT,timeout=120)
            if check.returncode:report['blocked'].append('GPU/import preflight failed')
            else:
                with (output/'training.log').open('w') as log:
                    run=subprocess.run([str(python),'-m','lerobot.scripts.lerobot_train','--config_path='+str((output/'train.json').resolve())],env=env,stdout=log,stderr=subprocess.STDOUT,timeout=1800)
                report['training_run']=True;report['pretrained_backward_passed']=run.returncode==0
                if run.returncode:report['blocked'].append('Training smoke failed; see training.log')
                else:
                    checkpoint=output/'training/checkpoints/last/pretrained_model'
                    m=read(Path(dataset)/'collection.json')
                    write(checkpoint/'cobot_data.json',dict(contract=m['contract'],track=m['track'],
                          training_collection_sha256=file_sha(Path(dataset)/'collection.json'),fps=20))
    if not execute and not report['blocked']:report['blocked'].append('Training not executed; rerun with --execute')
    report['passed']=report['data_loader_passed'] and report['pretrained_backward_passed']
    write(output/'report.json',report)
    return report
