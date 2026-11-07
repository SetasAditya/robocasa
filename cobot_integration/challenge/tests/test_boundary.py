import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from cobot_integration.challenge.catalog import VERSION, recipe
from cobot_integration.challenge.runner import apply, digest, evaluate, load, prepare, source_hash

class BoundaryTests(unittest.TestCase):
    def test_action_input_is_not_mutated_by_native_controller(self):
        class Env:
            action_spec=(np.array([-1.]),np.array([1.]))
            def step(self,a):a[:]=.7;return None
        a=np.array([.1]);apply(Env(),a)
        np.testing.assert_array_equal(a,[.1])
        for bad in ([2.],[float('nan')],[0.,0.]):
            with self.assertRaises(ValueError):apply(Env(),bad)

    def test_manipulation_and_recovery_need_physical_initialization(self):
        with tempfile.TemporaryDirectory() as d:
            for r in (recipe('PlateRackInsertion',mode='manipulation'),recipe('ImperfectGraspRecovery')):
                p=Path(d)/r['profile']
                with self.assertRaises(ValueError):prepare(p,r)
                self.assertFalse(p.exists())

    def test_seeds_cannot_leak_between_cells(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):prepare(Path(d)/'new',recipe('PlateRackInsertion'),seed=0)

    def test_asset_and_scene_tampering_fail_before_simulator_import(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);asset=p/'mesh.obj';asset.write_text('old')
            r=dict(version=VERSION,source_sha256=source_hash(),asset_files={str(asset):digest(asset)},files={})
            (p/'scene.json').write_text(json.dumps(r))
            asset.write_text('changed')
            with self.assertRaisesRegex(ValueError,'asset changed'):load(p)
            r['asset_files']={};r['files']={'state.npy':'bad'}
            (p/'state.npy').write_text('changed');(p/'scene.json').write_text(json.dumps(r))
            with self.assertRaisesRegex(ValueError,'artifact changed'):load(p)

    def test_formal_evaluation_rejects_practice_and_unqualified_scenes(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);record=dict(split='practice')
            (p/'scene.json').write_text(json.dumps(record))
            with self.assertRaisesRegex(ValueError,'practice'):evaluate(p,'unused',p/'output')
            record['split']='formal';(p/'scene.json').write_text(json.dumps(record))
            (p/'qualification.json').write_text(json.dumps(dict(qualified=False)))
            with self.assertRaisesRegex(ValueError,'qualification'):evaluate(p,'unused',p/'output')
