import copy,json,tempfile,unittest
from pathlib import Path
from cobot_scenarios.core import generate,validate,Ledger,write_new
from cobot_scenarios.spec import CATEGORIES,TASKS

def registry():
 return {'objects':{c:{s:[{'path':f'{c}/{s}/{i}/model.xml','sha256':'fake','qualification':'pending'} for i in range(n)] for s,n in [('train',3),('validation',1),('test',2)]} for c in CATEGORIES}}
class CurriculumTests(unittest.TestCase):
 def setUp(self):self.registry=registry();self.rows=generate(self.registry)
 def test_exact_quota_and_seed_counts(self):
  self.assertEqual(len(self.rows),1000)
  self.assertEqual(sum(r['collection_kind']=='human_seed' for r in self.rows),96)
  for task,spec in TASKS.items():
   for split,n in zip(('train','validation'),spec['counts']):self.assertEqual(sum(r['task']==task and r['split']==split for r in self.rows),n)
 def test_incomplete_curriculum_rejected(self):
  with self.assertRaises(ValueError):validate(self.rows[:-1],self.registry)
 def test_reproducible_and_disjoint(self):
  self.assertEqual(self.rows,generate(self.registry));self.assertTrue(validate(self.rows,self.registry))
  for split in ('train','validation'):
   other='validation' if split=='train' else 'train'
   self.assertFalse({r['seed_slot'] for r in self.rows if r['split']==split}&{r['seed_slot'] for r in self.rows if r['split']==other})
 def test_tampering_and_asset_leakage(self):
  changed=copy.deepcopy(self.rows);changed[0]['layout']=2
  with self.assertRaises(ValueError):validate(changed,self.registry)
  leaked=copy.deepcopy(self.registry);leaked['objects']['can']['test'][0]=leaked['objects']['can']['train'][0]
  with self.assertRaises(ValueError):validate(self.rows,leaked)
 def test_immutable_manifest(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'one.json';write_new(p,{'a':1});write_new(p,{'a':1})
   with self.assertRaises(ValueError):write_new(p,{'a':2})
 def test_claim_conflict_and_evidence(self):
  with tempfile.TemporaryDirectory() as d:
   l=Ledger(str(Path(d)/'db'));self.addCleanup(l.db.close);l.initialize(self.rows);id=self.rows[0]['scenario_id']
   l.update(id,'alice',0,'assigned')
   with self.assertRaises(ValueError):l.update(id,'bob',0,'assigned')
   with self.assertRaises(ValueError):l.update(id,'alice',1,'accepted')
   l.update(id,'alice',1,'recorded','demo.hdf5')
   with self.assertRaises(ValueError):l.update(id,'alice',2,'accepted',lineage='seed')
   evidence=dict(scenario_id=id,physical_success=True,replay=True,sensor_alignment=True,collision_review=True,instruction_review=True)
   l.update(id,'alice',2,'accepted',lineage='seed',evidence=evidence)
 def test_lineage_limit_and_split(self):
  with tempfile.TemporaryDirectory() as d:
   l=Ledger(str(Path(d)/'db'));self.addCleanup(l.db.close);l.initialize(self.rows)
   train=[r for r in self.rows if r['split']=='train'][:16]
   for i,r in enumerate(train):
    id=r['scenario_id'];l.update(id,'a',0,'assigned');l.update(id,'a',1,'recorded','demo')
    ev=dict(scenario_id=id,physical_success=True,replay=True,sensor_alignment=True,collision_review=True,instruction_review=True)
    if i<15:l.update(id,'a',2,'accepted',lineage='shared',evidence=ev)
    else:
     with self.assertRaises(ValueError):l.update(id,'a',2,'accepted',lineage='shared',evidence=ev)
   r=next(r for r in self.rows if r['split']=='validation');id=r['scenario_id']
   l.update(id,'a',0,'assigned');l.update(id,'a',1,'recorded','demo');ev['scenario_id']=id
   with self.assertRaises(ValueError):l.update(id,'a',2,'accepted',lineage='shared',evidence=ev)
if __name__=='__main__':unittest.main()
