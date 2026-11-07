import json
import unittest
from cobot_integration.challenge.catalog import PROFILES, DIFFICULTIES, MODES, VERSION, recipe, plan
from cobot_integration.challenge.scoring import ScoreState


class ContractTests(unittest.TestCase):
    def test_disjoint_seed_spaces(self):
        seen = set()
        for t in PROFILES:
            for d in DIFFICULTIES:
                for m in MODES:
                    for split in ('development','practice','formal'):
                        seeds = {s['seed'] for s in plan(t,d,m,split)['scenes']}
                        self.assertFalse(seeds & seen)
                        seen |= seeds

    def test_no_unqualified_profile_claims_success(self):
        for t in PROFILES:
            r = recipe(t)
            self.assertEqual(r['status'], 'experimental')
            self.assertFalse(r['physical_validation'])
            self.assertEqual(r['layout'], 100)
            self.assertEqual(r['version'], VERSION)

    def test_recipe_independence_and_validation(self):
        r=recipe('PlateRackInsertion'); r['parameters']['clearance_m']=1
        self.assertEqual(recipe('PlateRackInsertion')['parameters']['clearance_m'], .03)
        for args in [('bad',),('PlateRackInsertion','extreme'),('PlateRackInsertion','easy','bad')]:
            with self.assertRaises(ValueError): recipe(*args)
        with self.assertRaises(ValueError): plan('PlateRackInsertion',count=10001)

    def test_one_second_stability_not_predicate_calls(self):
        s=ScoreState()
        self.assertFalse(s.advance(0.,goal=True))
        for _ in range(100): self.assertFalse(s.advance(0.,goal=True))
        self.assertFalse(s.advance(.99,goal=True))
        self.assertTrue(s.advance(1.,goal=True))
        self.assertFalse(s.advance(1.05,goal=False))

    def test_protected_violation_is_latched(self):
        s=ScoreState()
        s.advance(0.,goal=False,protected=False)
        s.advance(.1,goal=True)
        self.assertFalse(s.advance(2.,goal=True))
        self.assertTrue(s.failed)

    def test_penetration_is_latched(self):
        s=ScoreState();s.advance(0.,goal=False,penetration=True)
        self.assertFalse(s.advance(2.,goal=True))

    def test_temporal_replay_and_coverage(self):
        s=ScoreState();s.advance(0.,goal=True,cells=[1,2],articulated=True)
        s.advance(.5,goal=True,cells=[3])
        saved=json.loads(json.dumps(s.snapshot()))
        other=ScoreState();other.restore(saved)
        self.assertEqual(s.advance(1.,goal=True),other.advance(1.,goal=True))
        self.assertEqual(s.snapshot(),other.snapshot())
        self.assertEqual(other.covered,{1,2,3})
        self.assertTrue(other.articulated)

    def test_time_requires_explicit_restore(self):
        s=ScoreState();s.advance(1.,goal=False)
        with self.assertRaises(ValueError):s.advance(0.,goal=False)
        with self.assertRaises(ValueError):s.advance(float('nan'),goal=False)
        with self.assertRaises(ValueError):s.restore({})


if __name__=='__main__':unittest.main()
