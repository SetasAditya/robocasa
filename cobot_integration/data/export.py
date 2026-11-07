"""Write and inspect real LeRobot v3 datasets; never upload to the Hub."""
import importlib.metadata
from pathlib import Path
import h5py
import numpy as np
from .manifest import VERSION, identity, write, read, file_sha
from .contract import features
from .validation import certified,decode

WRITER_VERSION='0.4.4'


def select(episodes,track,split):
    selected=[];seen=set();contract=None;lineage_splits={}
    for p in episodes:
        p=Path(p).resolve();m,v=certified(p);r=m['request']
        if r['track']!=track:raise ValueError('Cannot mix experiment tracks')
        previous=lineage_splits.setdefault(r['lineage'],r['split'])
        if previous!=r['split']:raise ValueError('Scene lineage leakage')
        if r['split']!=split:raise ValueError('Explicit split selection required; do not silently filter episodes')
        if split!='test' and not v['training_eligible']:raise ValueError('Episode is evaluation-only')
        if m['capture_sha256'] in seen:raise ValueError('Duplicate trajectory')
        seen.add(m['capture_sha256'])
        if contract is not None and m['contract']!=contract:raise ValueError('Incompatible controller/camera contracts')
        contract=m['contract'];selected.append((p,m,v))
    if not selected:raise ValueError('No eligible episodes')
    return selected,contract


def export(episodes,output,track,split,repo_id='local/cobot-robocasa'):
    if importlib.metadata.version('lerobot')!=WRITER_VERSION:raise RuntimeError('Use isolated lerobot==0.4.4 writer')
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    selected,spec=select(episodes,track,split);output=Path(output)
    if output.exists():raise FileExistsError(output)
    ds=LeRobotDataset.create(repo_id=repo_id,root=output,fps=20,robot_type=spec['robot'],
        features=features(spec),use_videos=True,video_backend='pyav',vcodec='h264',image_writer_threads=2)
    provenance=[]
    for i,(path,m,v) in enumerate(selected):
        with h5py.File(path/'capture.h5') as f:
            for j in range(m['frames']):
                ds.add_frame({'observation.images.kinect':decode(f['kinect'][j]),
                    'observation.images.wrist':decode(f['wrist'][j]),'observation.state':f['state'][j],
                    'action':f['action'][j],'task':m['instruction']})
            ds.save_episode(parallel_encoding=False)
        provenance.append(dict(episode_index=i,path=str(path),request=m['request'],frames=m['frames'],
            capture_sha256=m['capture_sha256'],episode_sha256=file_sha(path/'episode.json'),
            validation_sha256=file_sha(path/'validation.json'),scene_id=m['scene_id']))
    ds.finalize()
    files={str(p.relative_to(output)):file_sha(p) for p in output.rglob('*') if p.is_file()}
    manifest=dict(version=VERSION,repo_id=repo_id,track=track,split=split,contract=spec,episodes=provenance,
        frames=sum(m['frames'] for _,m,_ in selected),writer_version=WRITER_VERSION,
        files=files,purpose='evaluation' if split=='test' else 'training' if split=='train' else 'validation')
    write(output/'collection.json',manifest)
    return manifest


def inspect(output):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    output=Path(output);m=read(output/'collection.json')
    for name,sha in m['files'].items():
        if file_sha(output/name)!=sha:raise ValueError('Dataset file changed: '+name)
    ds=LeRobotDataset(m['repo_id'],root=output,video_backend='pyav')
    if len(ds)!=m['frames'] or ds.fps!=20:raise ValueError('Dataset row count or clock changed')
    cursor=0;max_rgb=0.
    for ep in m['episodes']:
        path=Path(ep['path']);meta,_=certified(path)
        if meta['capture_sha256']!=ep['capture_sha256']:raise ValueError('Export source changed')
        with h5py.File(path/'capture.h5') as f:
            for j in range(ep['frames']):
                row=ds[cursor+j]
                if row['task']!=meta['instruction']:raise ValueError('Exported language instruction differs')
                for field,key in [('action','action'),('observation.state','state')]:
                    value=row[field].numpy() if hasattr(row[field],'numpy') else np.asarray(row[field])
                    if not np.allclose(value,f[key][j],atol=1e-6,rtol=0):raise ValueError('Exported numeric row differs')
                if int(row['episode_index'])!=ep['episode_index']:raise ValueError('Episode boundary differs')
                if j in {0,ep['frames']//2,ep['frames']-1}:
                    for camera in ('kinect','wrist'):
                        value=row['observation.images.'+camera]
                        value=value.numpy() if hasattr(value,'numpy') else np.asarray(value)
                        value=np.moveaxis(value,0,-1)*255
                        max_rgb=max(max_rgb,float(np.abs(value-decode(f[camera][j])).mean()))
        cursor+=ep['frames']
    if max_rgb>12:raise ValueError('Decoded dataset video differs excessively')
    result=dict(passed=True,frames=cursor,episodes=len(m['episodes']),sampled_rgb_mae=max_rgb,
                collection_sha256=file_sha(output/'collection.json'))
    write(output/'inspection.json',result,overwrite=True)
    return result
