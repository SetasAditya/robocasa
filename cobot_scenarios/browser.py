"""Loopback-only browser collector. One synchronous simulator owner, one action per request.
No queued key events: client waits for each response; disconnect pauses simulation.
"""
import base64,io,json,secrets,time
from http.server import BaseHTTPRequestHandler,HTTPServer
from pathlib import Path
from .core import read,write_new
from .runtime import restore,collisions

class Session:
    def __init__(self,bundle,output):
        self.bundle=Path(bundle);self.output=Path(output);self.env=None;self.recorder=None;self.reset()
    def reset(self):
        if self.recorder is not None:raise ValueError('Save or discard the active recording first')
        if self.env is not None:self.env.close()
        self.env,self.resolved=restore(self.bundle,True);self.steps=0;self.closed=False;self._target()
    def _target(self):
        from scipy.spatial.transform import Rotation
        r=self.env.robots[0];sid=r.eef_site_id['right']
        self.xyz=self.env.sim.data.site_xpos[sid].copy();self.rot=Rotation.from_matrix(self.env.sim.data.site_xmat[sid].reshape(3,3)).as_rotvec()
    def start(self):
        if self.recorder is not None:raise ValueError('Already recording')
        if self.resolved['qualification']['initialization']!='passed':raise ValueError('Initial collision screen failed; use another qualified bundle')
        if self.steps:raise ValueError('Reset before recording to preserve the frozen initial state')
        from kitchen_recording import KitchenRecorder
        from .runtime import snapshot
        self.folder=self.output/(self.resolved['scenario_id']+'-'+time.strftime('%Y%m%d-%H%M%S')+'-'+secrets.token_hex(3))
        # Preserve exact frozen XML during recorder canonicalization.
        original=self.env.model.get_xml
        self.env.model.get_xml=lambda:(self.bundle/'model.xml').read_text()
        try:self.recorder=KitchenRecorder(self.env,snapshot,lambda e:e._check_success(),self.resolved['config'],self.folder,task_metadata=dict(scenario_id=self.resolved['scenario_id'],split=self.resolved['request']['split'],seed_slot=self.resolved['request']['seed_slot'],instruction=self.resolved['instruction'],raw_action_contract='world_absolute_tcp_rotvec_base_controller_gripper_mode'))
        finally:self.env.model.get_xml=original
        self.transforms=[];self.penetrations=[]
    def step(self,request):
        import numpy as np
        from scipy.spatial.transform import Rotation
        move=np.asarray(request.get('move',[0,0,0]),dtype=float);turn=np.asarray(request.get('turn',[0,0,0]),dtype=float)
        if move.shape!=(3,) or turn.shape!=(3,) or not np.isfinite(np.r_[move,turn]).all():raise ValueError('Invalid input')
        move=np.clip(move,-1,1);turn=np.clip(turn,-1,1)
        mode=request.get('mode','arm');base=np.zeros(3)
        R=self.env.sim.data.get_body_xmat('mobilebase0_base').copy()
        if mode=='base':
            self._target()
            # Convert body-relative velocity into the legacy controller frame.
            world=R[:2,:2]@(move[:2]*.15)
            names=['mobilebase0_joint_mobile_forward','mobilebase0_joint_mobile_side']
            axes=np.stack([self.env.sim.data.xaxis[self.env.sim.model.joint_name2id(n)][:2] for n in names],axis=1)
            c=self.env.robots[0].part_controllers['base'];_,ori=c.get_base_pose()
            theta=np.arctan2(ori[1,0],ori[0,0])-np.arctan2(c.init_ori[1,0],c.init_ori[0,0])
            base[:2]=np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])@np.linalg.solve(axes,world)
            base[2]=turn[2]*.3
        elif mode=='arm':
            self.xyz+=R@move*.005
            self.rot=(Rotation.from_rotvec(R@turn*.02)*Rotation.from_rotvec(self.rot)).as_rotvec()
        else:raise ValueError('Unknown mode')
        if 'closed' in request:self.closed=bool(request['closed'])
        action=self.env.robots[0].create_action_vector({'right':np.r_[self.xyz,self.rot], 'right_gripper':[1. if self.closed else -1.], 'base':base, 'base_mode':-1})
        stage=request.get('stage','approach')
        if stage not in self.resolved['request']['stages']:raise ValueError('Unknown stage')
        if self.recorder:
            self.transforms.append(dict(base_xyz=self.env.sim.data.get_body_xpos('mobilebase0_base').tolist(),base_rotation=R.tolist(),tcp_xyz=self.env.sim.data.site_xpos[self.env.robots[0].eef_site_id['right']].tolist()))
            self.recorder.step(action,stage);self.penetrations.extend(collisions(self.env))
        else:self.env.step(action.copy())
        self.steps+=1
    def save(self):
        if not self.recorder or not self.recorder.actions:raise ValueError('No recorded actions')
        path=self.recorder.save(self.folder/'demo.hdf5');self.recorder=None
        write_new(self.folder/'scenario.json',self.resolved)
        write_new(self.folder/'transforms.json',self.transforms)
        write_new(self.folder/'collection.json',dict(scenario_id=self.resolved['scenario_id'],physical_success=bool(self.env._check_success()),penetrations=self.penetrations,status='recorded_unreviewed'))
        return str(path)
    def discard(self):
        if self.recorder:
            self.recorder.sensors.close();self.recorder=None
            write_new(self.folder/'discarded.json',dict(reason='collector_discarded'))
        self.reset()
    def status(self):
        from PIL import Image
        from kitchen_recording import render_camera
        images={}
        for name in ('kinect','wrist'):
            buf=io.BytesIO();Image.fromarray(render_camera(self.env,name)).save(buf,format='JPEG',quality=75)
            images[name]=base64.b64encode(buf.getvalue()).decode()
        return dict(images=images,scenario_id=self.resolved['scenario_id'],instruction=self.resolved['instruction'],stages=self.resolved['request']['stages'],steps=self.steps,recording=self.recorder is not None,success=bool(self.env._check_success()))

def replay(bundle,recording):
    import h5py,numpy as np
    from kitchen_recording import audit_sensor_alignment
    env,r=restore(bundle,False)
    try:
        with h5py.File(recording) as f:
            d=f['data/demo_0'];meta=json.loads(d.attrs['task_metadata'])
            if meta['scenario_id']!=r['scenario_id']:raise ValueError('Recording belongs to another scenario')
            errors=[]
            for state,action in zip(d['states'],d['actions']):
                errors.append(float(np.max(np.abs(env.sim.get_state().flatten()-state))))
                env.step(np.asarray(action).copy())
            errors.append(float(np.max(np.abs(env.sim.get_state().flatten()-d['final_state'][:]))))
            report=dict(scenario_id=r['scenario_id'],replay=max(errors,default=float('inf'))<1e-5,max_state_error=max(errors),physical_success=bool(env._check_success()),sensor_alignment=audit_sensor_alignment(recording)['passed'],collision_review=False,instruction_review=False)
            return report
    finally:env.close()

def serve(bundle,output,port):
    session=Session(bundle,output);token=secrets.token_urlsafe(24)
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def do_GET(self):
            if self.path!='/':self.send_error(404);return
            body=(Path(__file__).with_name('console.html').read_text().replace('__TOKEN__',token)).encode()
            self.send_response(200);self.send_header('Content-Type','text/html');self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body)
        def do_POST(self):
            if self.headers.get('X-Session-Token')!=token:self.send_error(403);return
            try:
                size=int(self.headers.get('Content-Length',0))
                if size>8192:raise ValueError('Request too large')
                request=json.loads(self.rfile.read(size));op=request.get('op')
                extra={}
                if op=='step':session.step(request)
                elif op=='start':session.start()
                elif op=='save':extra['saved']=session.save()
                elif op=='discard':session.discard()
                elif op=='reset':session.reset()
                elif op!='status':raise ValueError('Unknown operation')
                body=json.dumps({**session.status(),**extra}).encode();code=200
            except Exception as e:body=json.dumps(dict(error=str(e))).encode();code=400
            self.send_response(code);self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body)
    server=HTTPServer(('127.0.0.1',port),Handler)
    print(f'Browser collector: http://127.0.0.1:{port} (use SSH forwarding)',flush=True)
    try:server.serve_forever()
    finally:
        if session.recorder:session.recorder.sensors.close()
        session.env.close();server.server_close()
