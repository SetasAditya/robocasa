import unittest
from cobot_scenarios.geometry import route, fitting_yaw
import math
class GeometryTests(unittest.TestCase):
 def test_appliance_yaw_preserves_feasible_request(self):
  self.assertEqual(fitting_yaw((.24,.24),(.344,.26),0),0)
 def test_appliance_yaw_rejects_impossible_rotation(self):
  requested=math.radians(252.19883774738128)
  actual=fitting_yaw((.24,.24),(.344,.26),requested)
  self.assertNotEqual(actual,requested)
  extent=.24*(abs(math.sin(actual))+abs(math.cos(actual)))
  self.assertLessEqual(extent+.002,.26)
  self.assertLess(abs(actual-requested),math.radians(20))
 def test_oversized_asset_fails_without_scaling(self):
  with self.assertRaisesRegex(ValueError,'cannot fit'):
   fitting_yaw((.4,.4),(.344,.26),0)
 def test_route_around_obstacle(self):
  path=route((0,0),(3,0),[(1,-.3,2,.3)],(-1,-2,4,2))
  self.assertIsNotNone(path)
  self.assertTrue(any(abs(y)>.7 for x,y in path))
 def test_blocked_start_and_sealed_wall(self):
  self.assertIsNone(route((0,0),(3,0),[(-.1,-.1,.1,.1)],(-1,-2,4,2)))
  self.assertIsNone(route((0,0),(3,0),[(1,-3,2,3)],(-1,-2,4,2)))
