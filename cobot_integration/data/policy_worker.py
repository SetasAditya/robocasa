"""Local π0.5 inference worker in the pinned training interpreter."""
import argparse
import os
from pathlib import Path
import socket
import numpy as np
from .manifest import read
from .protocol import receive,send


def main():
    p=argparse.ArgumentParser();p.add_argument('--socket',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True)
    a=p.parse_args()
    import torch
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy
    from lerobot.policies.factory import make_pre_post_processors
    identity=read(a.checkpoint/'cobot_data.json')
    policy=PI05Policy.from_pretrained(a.checkpoint,local_files_only=True)
    policy.eval()
    pre,post=make_pre_post_processors(policy.config,pretrained_path=a.checkpoint,
        preprocessor_overrides={'device_processor':{'device':policy.config.device}})
    policy.reset()
    with socket.socket(socket.AF_UNIX,socket.SOCK_STREAM) as server:
        server.bind(str(a.socket));os.chmod(a.socket,0o600);server.listen(1)
        conn,_=server.accept()
        with conn:
            send(conn,dict(ok=True,contract=identity['contract']))
            while True:
                meta,arrays=receive(conn)
                if meta.get('command')=='close':break
                if meta.get('command')!='infer':raise ValueError('Unknown policy request')
                batch={}
                for key,value in arrays.items():
                    tensor=torch.from_numpy(value)
                    if key.startswith('observation.images.'):
                        tensor=tensor.permute(2,0,1).float()/255.
                    batch[key]=tensor.unsqueeze(0)
                batch['task']=[meta['instruction']]
                # Recompute a chunk each request; the simulator executes only its first five actions.
                with torch.inference_mode():
                    chunk=post(policy.predict_action_chunk(pre(batch))).squeeze(0).detach().cpu().numpy()
                if chunk.ndim!=2 or chunk.shape[1]!=identity['contract']['action_dim'] or not np.isfinite(chunk).all():
                    raise ValueError('Invalid predicted action chunk')
                send(conn,dict(ok=True),{'action':chunk})


if __name__=='__main__':main()
