import unittest
from cobot_integration.challenge.goals import released_rest, rack_seated, bowl_seated, validate_geometry, tray_guides

class GoalTests(unittest.TestCase):
    def test_held_and_moving_targets_fail(self):
        args=dict(in_destination=True,released=True,linear_speed=0.,angular_speed=0.)
        self.assertTrue(released_rest(**args))
        for changes in (dict(released=False),dict(in_destination=False),dict(linear_speed=.1),dict(angular_speed=1.),dict(linear_speed=float('nan'))):
            self.assertFalse(released_rest(**(args|changes)))

    def test_wrong_slot_hovering_and_orientation_fail(self):
        args=dict(delta=[0.,0.,0.],half_extents=[.01,.1,.1],orientation_alignment=1.,rack_contact=True)
        self.assertTrue(rack_seated(**args))
        for changes in (dict(delta=[.02,0,0]),dict(rack_contact=False),dict(orientation_alignment=0.),dict(delta=[float('nan'),0,0])):
            self.assertFalse(rack_seated(**(args|changes)))

    def test_bowl_requires_seating_support_and_release(self):
        args=dict(height=.04,height_interval=[.03,.05],xy_distance=.001,tolerance=.02,
                  target_upright=True,support_upright=True,contact=True,both_inside=True,
                  support_quiet=True,support_released=True)
        self.assertTrue(bowl_seated(**args))
        for key in ('target_upright','support_upright','contact','both_inside','support_quiet','support_released'):
            self.assertFalse(bowl_seated(**(args|{key:False})))
        self.assertFalse(bowl_seated(**(args|dict(height=.15))))
        self.assertFalse(bowl_seated(**(args|dict(xy_distance=.03))))

    def test_stacking_and_nesting_geometry_are_distinct(self):
        valid=dict(seated_height_interval_m=[.03,.05],relationship='nest')
        validate_geometry('nest',valid)
        with self.assertRaises(ValueError): validate_geometry('stack',valid)
        with self.assertRaises(ValueError): validate_geometry('nest',dict(seated_height_interval_m=[.05,.03],relationship='nest'))
        with self.assertRaises(ValueError): validate_geometry('plate',dict(slot=dict(center=[0,0,0],half_extents=[0,.1,.1])))

    def test_tray_stop_and_clearance_match_real_footprint(self):
        parts=tray_guides([.30,.40,.03],.02)
        left,right,stop=parts
        self.assertAlmostEqual(right[0][0]-right[1][0]-(left[0][0]+left[1][0]),.32)
        self.assertAlmostEqual(stop[0][1]-stop[1][1],.20)
        with self.assertRaises(ValueError):tray_guides([0,.40,.03],.02)
