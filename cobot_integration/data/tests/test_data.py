import ast,io,json,tempfile,types,unittest
from pathlib import Path
import h5py,numpy as np
from PIL import Image
from cobot_integration.data.manifest import plan,validate_plan,check_request,file_sha,write,read,VERSION
from cobot_integration.data.contract import describe,action
from cobot_integration.data.task_state import pack,unpack,declared_fields
from cobot_integration.data.validation import inspect_raw,certified
from cobot_integration.data.export import select
from cobot_integration.data.training import recipe
ROOT=Path(__file__).resolve().parents[3]
class Kitchen:pass
class CounterTask(Kitchen):
    def __init__(self):self.counter=0;self.points=[]
    def update(self):self.counter+=1;self.points.append(np.array([1.,2.]))
def spec():
    c=types.SimpleNamespace(input_type='delta',input_ref_frame='base',kp=np.ones(6)*300,output_min=-np.ones(6),output_max=np.ones(6))
    r=types.SimpleNamespace(part_controllers={'right':c},composite_controller=types.SimpleNamespace(_action_split_indexes={'right':(0,6),'right_gripper':(6,7),'base':(7,10),'base_mode':(10,11)}))
    return describe(types.SimpleNamespace(robots=[r],control_freq=20,action_dim=11,action_spec=(-np.ones(11),np.ones(11))))
def episode(path):
    path.mkdir();r=plan('transfer',tasks=['OpenDrawer'],count=1)['requests'][0];s=spec();n=3
    buf=io.BytesIO();Image.fromarray(np.full((360,640,3),100,dtype=np.uint8)).save(buf,format='JPEG');image=np.frombuffer(buf.getvalue(),dtype='uint8')
    with h5py.File(path/'capture.h5','w') as f:
        for key,value in dict(sim_state=np.zeros((n,5)),state=np.zeros((n,18)),action=np.zeros((n,11)),timestamp=np.arange(n)/20,success=np.ones(n,dtype=bool),contact_violation=np.zeros(n,dtype=bool),final_state=np.zeros(5)).items():f[key]=value
        for key in ('phase','task_state'):f.create_dataset(key,data=['{}']*n,dtype=h5py.string_dtype())
        for key in ('kinect','wrist'):
            ds=f.create_dataset(key,(n,),dtype=h5py.vlen_dtype(np.dtype('uint8')))
            for i in range(n):ds[i]=image
    m=dict(version=VERSION,request=r,frames=n,contract=s,capture_sha256=file_sha(path/'capture.h5'),scene_id='unit-test-scene',source_sha256='unit-test-source',success=True,status='recorded')
    write(path/'episode.json',m)
    write(path/'validation.json',dict(passed=True,checks={k:True for k in ('recording_completed','recorded_success','initial_scene','scene_binding','timing','action_replay','state18_replay','task_history_replay','cameras','contacts','terminal_success','geometry_review')},training_eligible=True,episode_sha256=file_sha(path/'episode.json'),capture_sha256=m['capture_sha256'],scene_id=m['scene_id'],source_sha256=m['source_sha256']))
class DataTests(unittest.TestCase):
    def test_split_layouts_and_seeds(self):
        p=plan('transfer',count=2);validate_plan(p);self.assertEqual(len(p['requests']),36)
        self.assertTrue(all(r['layout']!=100 for r in p['requests'] if r['split']!='test'))
        self.assertTrue(all(r['layout']==100 for r in p['requests'] if r['split']=='test'))
        self.assertEqual(len({r['seed'] for r in p['requests']}),36)
    def test_full_layout_style_product_and_custom_holdouts(self):
        p=plan('transfer',tasks=['OpenDrawer'],count=8)
        pairs={(r['layout'],r['style']) for r in p['requests'] if r['split']=='train'}
        self.assertEqual(len(pairs),8)
        config={'layouts':{'train':[16],'validation':[24],'test':[100]},'styles':{'train':[11],'validation':[33],'test':[1]}}
        self.assertEqual(plan('transfer',tasks=['OpenDrawer'],count=1,kitchen_config=config)['requests'][0]['layout'],16)
        config['layouts']['train']=[100]
        with self.assertRaises(ValueError):plan('transfer',kitchen_config=config)
    def test_solver_boundary_does_not_set_robot_pose(self):
        from cobot_integration.data.runtime import begin_step
        data=types.SimpleNamespace(qacc_warmstart=np.ones(3),qpos=np.array([1.,2.]),qvel=np.array([3.,4.]))
        calls=[];ctrl=types.SimpleNamespace(update=lambda force:calls.append(force))
        env=types.SimpleNamespace(sim=types.SimpleNamespace(data=data),robots=[types.SimpleNamespace(part_controllers={'right':ctrl})])
        begin_step(env)
        np.testing.assert_array_equal(data.qacc_warmstart,np.zeros(3))
        np.testing.assert_array_equal(data.qpos,[1.,2.]);np.testing.assert_array_equal(data.qvel,[3.,4.])
        self.assertEqual(calls,[True])
    def test_local_policy_transport_round_trip(self):
        import struct
        from cobot_integration.data.protocol import send,receive,MAX_BYTES
        class Pipe:
            data=b''
            def sendall(self,b):self.data+=b
            def recv(self,n):out=self.data[:n];self.data=self.data[n:];return out
        pipe=Pipe();send(pipe,{'command':'infer'},{'state':np.arange(18,dtype=np.float32)})
        meta,arrays=receive(pipe)
        self.assertEqual(meta,{'command':'infer'});np.testing.assert_array_equal(arrays['state'],np.arange(18))
        pipe.data=struct.pack('!I',MAX_BYTES+1)
        with self.assertRaises(ValueError):receive(pipe)
        with self.assertRaises(ValueError):send(Pipe(),{}, {'bad':np.array([object()],dtype=object)})
    def test_challenge_track(self):
        p=plan('challenge',count=1,difficulties=['easy','hard']);self.assertEqual(len(p['requests']),60)
        self.assertTrue(all(r['layout']==100 for r in p['requests']))
    def test_request_tampering(self):
        r=plan('transfer',count=1)['requests'][0]
        for key,value in [('layout',100),('seed',0),('split','test'),('asset_split','target')]:
            bad=dict(r);bad[key]=value
            with self.assertRaises(ValueError):check_request(bad)
        with self.assertRaises(ValueError):plan('challenge',modes=['teleport'])
        with self.assertRaises(ValueError):plan('transfer',count=0)
    def test_native_commands_keep_mode_and_copy(self):
        s=spec();a=np.zeros(11);a[10]=-1;b=action(a,s);b[0]=1
        self.assertEqual(a[0],0);self.assertEqual(b[10],-1);self.assertEqual(s['action_dim'],11)
        for bad in (np.zeros(10),np.full(11,np.nan),np.full(11,2)):
            with self.assertRaises(ValueError):action(bad,s)
    def test_pickle_free_task_history(self):
        v={'points':[np.array([1.,2.])],'count':3,'tuple':(1,2),'covered':{2,3}}
        restored=unpack(json.loads(json.dumps(pack(v))))
        np.testing.assert_array_equal(restored['points'][0],v['points'][0])
        self.assertEqual(restored['covered'],{2,3});self.assertEqual(restored['tuple'],(1,2))
        self.assertEqual(set(declared_fields(CounterTask)),{'counter','points'})
        with self.assertRaises(TypeError):pack(object())
    def test_capture_hash_and_time_alignment(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'ep';episode(p);self.assertEqual(inspect_raw(p)['frames'],3)
            with h5py.File(p/'capture.h5','r+') as f:f['timestamp'][1]=0
            with self.assertRaisesRegex(ValueError,'hash'):inspect_raw(p)
            m=read(p/'episode.json');m['capture_sha256']=file_sha(p/'capture.h5');write(p/'episode.json',m,overwrite=True)
            with self.assertRaisesRegex(ValueError,'timestamps'):inspect_raw(p)
    def test_stale_certificate(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'ep';episode(p);certified(p)
            m=read(p/'episode.json');m['instruction']='changed';write(p/'episode.json',m,overwrite=True)
            with self.assertRaisesRegex(ValueError,'Stale'):certified(p)
    def test_export_cannot_mix_partitions_or_tracks(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'ep';episode(p)
            with self.assertRaisesRegex(ValueError,'tracks'):select([p],'challenge','train')
            with self.assertRaisesRegex(ValueError,'split'):select([p],'transfer','validation')
            with self.assertRaisesRegex(ValueError,'Duplicate'):select([p,p],'transfer','train')
    def test_training_cannot_use_test_partition(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);write(p/'collection.json',dict(split='test',purpose='evaluation'));write(p/'inspection.json',dict(passed=True))
            with self.assertRaisesRegex(ValueError,'training partition'):recipe(p,p/'out',p,p)
    def test_native_teleop_hooks(self):
        tree=ast.parse((ROOT/'robocasa/scripts/collect_demos.py').read_text())
        functions=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('is_empty_input_spacemouse','collect_human_trajectory')]
        class Wall:
            def __init__(self,e):pass
            def consume_pending(self,render):return False
        ns=dict(np=np,time=__import__('time'),deepcopy=__import__('copy').deepcopy,DataCollectionWrapper=type('Wrapper',(),{}),EnclosingWallHotkeyHandler=Wall,colored=lambda x,*_:x)
        exec(compile(ast.Module(body=functions,type_ignores=[]),'native_teleop','exec'),ns)
        class Robot:
            arms=['right'];gripper={'right':types.SimpleNamespace(dof=1)};part_controllers={'right':types.SimpleNamespace(input_type='delta')}
            def create_action_vector(self,p):return np.r_[p.get('right',np.zeros(6)),p.get('right_gripper',[0]),p.get('base',[0,0,0]),p.get('base_mode',-1)]
        class Env:
            action_dim=11;robots=[Robot()];ep_directory='test';closed=False;resets=0
            def reset(self):self.resets+=1
            def get_ep_meta(self):return {'lang':'test'}
            def _check_success(self):return False
            def close(self):self.closed=True
            def render(self):pass
        class Device:
            active_robot=0;active_arm='right'
            def start_control(self):pass
            def input2action(self,mirror_actions):return {'right_delta':np.ones(6)*.1,'right_gripper':[1.],'base':np.zeros(3),'base_mode':1}
        e=Env();calls=[]
        def step(a):calls.append(a.copy());return {},0,False,{}
        path,discard=ns['collect_human_trajectory'](e,Device(),'right','single',False,render=False,print_info=False,reset_fn=lambda:None,step_fn=step,warmup_steps=0,close_env=False,max_steps=2)
        self.assertEqual(len(calls),2);self.assertEqual(calls[0][-1],1);self.assertEqual(e.resets,0);self.assertFalse(e.closed);self.assertTrue(discard);self.assertEqual(path,'test')
        original=Env();direct=[]
        original.step=lambda a:(direct.append(a.copy()) or ({},0,False,{}))
        ns['collect_human_trajectory'](original,Device(),'right','single',False,render=False,print_info=False,max_steps=2)
        self.assertEqual(original.resets,1);self.assertTrue(original.closed);self.assertEqual(len(direct),3)

if __name__=='__main__':unittest.main()
