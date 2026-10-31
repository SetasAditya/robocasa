"""Regression tests for metric geometry, source conversion and safe placement."""
import unittest, json, copy, xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
import mujoco
from robocasa.models.scenes.kitchen_arena import KitchenArena
from robocasa.models.scenes.scene_registry import get_layout_path, unpack_layout_ids, LayoutType
from robocasa.models.fixtures.fixture import FixtureType
from robocasa.models.fixtures.fixture_utils import fixture_is_type
from . import ROOT

class LabAptTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.arena = KitchenArena(100, 1, rng=np.random.default_rng(7))
        cls.fx = cls.arena.fixtures

    def test_registration_is_explicit(self):
        self.assertEqual(unpack_layout_ids('LAB_APT'), [100])
        self.assertEqual(get_layout_path(100), get_layout_path('lab_apt'))
        self.assertNotIn(100, unpack_layout_ids(LayoutType.TRAIN))
        self.assertNotIn(100, unpack_layout_ids(LayoutType.TEST))

    def test_fixed_geometry_and_style(self):
        other = KitchenArena(100, 12, rng=np.random.default_rng(7))
        for name, f in self.fx.items():
            np.testing.assert_allclose(f.pos, other.fixtures[name].pos, atol=1e-10)
            np.testing.assert_allclose(f.size, other.fixtures[name].size, atol=1e-10)
            self.assertEqual(ET.tostring(f.asset), ET.tostring(other.fixtures[name].asset))

    def test_lower_cabinets_and_dining_semantics(self):
        self.assertTrue(fixture_is_type(self.fx['left_base_labapt'], FixtureType.CABINET_SINGLE_DOOR))
        self.assertTrue(fixture_is_type(self.fx['sink_cabinet_labapt'], FixtureType.CABINET_DOUBLE_DOOR))
        self.assertTrue(fixture_is_type(self.fx['dining_black_table_labapt'], FixtureType.DINING_COUNTER))
        self.assertFalse(self.fx['wall_shelf_1_labapt'].get_reset_regions(z_range=(0.05, 1.5)))
        self.assertTrue(self.fx['wall_shelf_0_labapt'].get_reset_regions())

    def test_support_regions_do_not_cross_appliances(self):
        for name in ('sink_counter_labapt', 'appliance_counter_labapt'):
            f = self.fx[name]
            regions = f.get_reset_regions(None)
            self.assertTrue(regions)
            for region in regions.values():
                x, y, _ = region['offset']
                w, d = region['size']
                r = (x - w / 2, y - d / 2, x + w / 2, y + d / 2)
                for b in f.blocked_rectangles:
                    overlap = min(r[2], b[2]) - max(r[0], b[0])
                    depth = min(r[3], b[3]) - max(r[1], b[1])
                    self.assertFalse(overlap > 1e-09 and depth > 1e-09, (name, r, b))

    def test_source_geometry_and_frame(self):
        arena = copy.deepcopy(self.arena)
        for f in arena.fixtures.values():
            arena.merge_assets(f)
            arena.worldbody.append(f.get_obj())
        ET.SubElement(arena.root, 'compiler', angle='radian')
        model = mujoco.MjModel.from_xml_string(arena.get_xml())
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        meta = json.loads((ROOT / 'robocasa/models/assets/fixtures/labapt/fridge/model.provenance.json').read_text())
        pos = np.asarray(self.fx['fridge_labapt'].pos)
        errors = []
        for r in meta['meshes']:
            gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, 'fridge_labapt_' + r['name'] + '_visual')
            mid = model.geom_dataid[gid]
            start = model.mesh_vertadr[mid]
            count = model.mesh_vertnum[mid]
            verts = np.asarray(model.mesh_vert[start:start + count])
            world = verts @ data.geom_xmat[gid].reshape(3, 3).T + data.geom_xpos[gid]
            want = np.array(r['bounds_source_m']) - [0, 0, 0.84] + pos
            errors.append(np.max(np.abs(np.stack([world.min(0), world.max(0)]) - want)))
        self.assertLess(max(errors), 0.001)
        manifest = json.loads(Path(get_layout_path(100)).with_suffix('.manifest.json').read_text())
        t = manifest['scan_to_layout']
        point = np.array([0.3393632741621875, -0.1883395174151129, -1.11])
        mapped = np.array(t['rotation_matrix']) @ point + np.array(t['translation'])
        np.testing.assert_allclose(mapped[:2], self.fx['dining_black_table_labapt'].pos[:2], atol=0.001)
        self.assertAlmostEqual(mapped[2], 0, places=8)

    def test_optional_rack_is_fixed(self):
        a = KitchenArena(100, 1, rng=np.random.default_rng(0), enable_fixtures=['dish_rack'])
        self.assertIn('dish_rack_labapt', a.fixtures)
        self.assertNotIn('dish_rack_labapt', self.fx)
        np.testing.assert_allclose(a.fixtures['dish_rack_labapt'].pos + a.fixtures['dish_rack_labapt'].origin_offset, [-1.25, -0.1, 1.01], atol=0.001)

    def test_episode_frame_provenance(self):
        from robocasa.models.scenes.labapt import episode_metadata
        meta = episode_metadata()
        self.assertTrue(meta['held_out'])
        self.assertEqual(meta['meters_per_unit'], 1)
        self.assertEqual(len(meta['asset_sha256']), 5)
        self.assertTrue(all((len(h) == 64 for h in meta['asset_sha256'].values())))
        self.assertEqual(np.asarray(meta['scan_to_layout']['rotation_matrix']).shape, (3, 3))

    def test_fixture_geometry_unchanged_by_room(self):
        path = Path(get_layout_path(100))
        old = json.loads(path.with_suffix('.fixture_baseline.json').read_text())
        current = json.loads(path.with_suffix('.manifest.json').read_text())
        self.assertEqual(old['fixtures'], current['fixtures'])
        self.assertEqual(old['scan_to_layout'], current['scan_to_layout'])

    def test_room_preserves_open_connections(self):
        spec = json.loads(Path(get_layout_path(100)).with_suffix('.walls.json').read_text())
        self.assertGreater(len(spec['walls']), 10)
        self.assertEqual({p['name'] for p in spec['openings'] if p['kind'].startswith('open')}, {'left_entry', 'front_connection'})
        divider = [s for s in spec['segments'] if s['plane'] == 'divider']
        self.assertAlmostEqual(max((s['z_range_m'][1] for s in divider)), 1.35)
        self.assertTrue(all((s['horizontal_range_m'][1] <= -1.76 for s in divider)))
        self.assertFalse(any(('front' in s['name'] for s in spec['segments'])))
        for plane in spec['plane_fits'].values():
            self.assertGreater(plane['inliers'], 50)
            self.assertLessEqual(plane['inlier_p95_error_m'], 0.025)

    def test_rotated_wall_boundary_queries(self):
        from robocasa.utils.object_utils import point_in_fixture
        spec = json.loads(Path(get_layout_path(100)).with_suffix('.walls.json').read_text())
        for s in spec['segments']:
            f = self.fx[s['name'] + '_room']
            pts = np.asarray(f.get_ext_sites(relative=False, all_points=True))
            self.assertEqual(pts.shape, (8, 3))
            self.assertTrue(np.isfinite(pts).all())
            self.assertTrue(point_in_fixture(np.asarray(f.pos), f, only_2d=True))
            outside = np.asarray(f.pos).copy()
            outside[:2] += np.asarray(s['outward_normal_xy']) * 0.3
            self.assertFalse(point_in_fixture(outside, f, only_2d=True))

    def test_existing_layouts_load(self):
        for layout in (2, 11):
            arena = KitchenArena(layout, 1, rng=np.random.default_rng(0))
            self.assertGreater(len(arena.fixtures), 10)
            self.assertTrue(all((not type(f).__name__.startswith('LabApt') for f in arena.fixtures.values())))
if __name__ == '__main__':
    unittest.main(verbosity=2)
