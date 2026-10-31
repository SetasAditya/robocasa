"""Geometry adaptations specific to the fixed LabApt layout.

No runtime state is forced. This runs while constructing MJCF, before compilation.
"""
import xml.etree.ElementTree as ET
import numpy as np
from robosuite.utils.mjcf_utils import array_to_string as a2s
from robocasa.models.fixtures.cabinets import Cabinet

def configure_fixtures(fixtures):
    for f in fixtures.values():
        if isinstance(f, Cabinet):
            f.task_selection_z_range = (0.05, 1.5)
        if f.name in ('microwave_labapt', 'dishwasher_labapt', 'dish_rack_labapt'):
            f.set_origin(np.asarray(f.pos).copy())
    for geom in fixtures['air_fryer_labapt'].get_obj().iter('geom'):
        if geom.get('group') == '1':
            geom.attrib.pop('material', None)
            geom.set('rgba', '0.025 0.025 0.025 1')
    import json
    from pathlib import Path
    from robocasa.models.scenes.scene_registry import get_layout_path
    wall_spec = json.loads(Path(get_layout_path(100)).with_suffix('.walls.json').read_text())
    for name, rgba in wall_spec['appearance_rgba'].items():
        f = fixtures[name + '_room']
        for geom in f.get_obj().iter('geom'):
            if geom.get('group') == '1':
                geom.attrib.pop('material', None)
                geom.set('rgba', a2s(rgba))
    panel = fixtures['pantry_labapt'].get_obj().find(".//geom[@name='pantry_labapt_right_door_door']")
    size = np.fromstring(panel.get('size'), sep=' ')
    size[0] -= 0.005
    panel.set('size', a2s(size))
    for name in ('left_base_labapt', 'drawer_3_labapt'):
        panel = fixtures[name].get_obj().find(".//geom[@name='" + name + "_door_door']")
        size = np.fromstring(panel.get('size'), sep=' ')
        size[1] -= 0.003
        panel.set('size', a2s(size))
    table = fixtures['dining_black_table_labapt']
    appearance = json.loads(Path(get_layout_path(100)).with_suffix('.manifest.json').read_text())['appearance']['dining_black_table']
    material = table.asset.find("material[@name='dining_black_table_labapt_counter_top']")
    for key in ('rgba', 'texrepeat'):
        material.set(key, a2s(appearance[key]))
    for key in ('reflectance', 'shininess'):
        material.set(key, str(appearance[key]))
    material.set('texuniform', 'false')
    for i, x in enumerate((-1.22, 1.22)):
        for visual in (False, True):
            attrs = dict(name=f'{table.name}_end_support_{i}' + ('_visual' if visual else ''), type='box', pos=a2s([x, 0, -0.02]), size=a2s([0.04, 0.36, 0.415]), group='1' if visual else '0', rgba='0.025 0.025 0.025 1' if visual else '0 0 0 0')
            if visual:
                attrs.update(contype='0', conaffinity='0', mass='0')
            ET.SubElement(table.get_obj(), 'geom', **attrs)
    occupied = [(-0.135, -0.1325, 0.135, 0.1325), (0.225, -0.215, 0.675, 0.235)]
    for name in ('sink_counter_labapt', 'appliance_counter_labapt'):
        f = fixtures[name]
        rectangles = []
        for x0, y0, x1, y1 in occupied:
            rectangles.append((x0 - f.pos[0] - 0.005, y0 - f.pos[1] - 0.005, x1 - f.pos[0] + 0.005, y1 - f.pos[1] + 0.005))
        if 'dish_rack_labapt' in fixtures:
            rack = fixtures['dish_rack_labapt']
            x, y = rack.pos[:2]
            w, d = rack.size[:2]
            rectangles.append((x - w / 2 - f.pos[0] - 0.005, y - d / 2 - f.pos[1] - 0.005, x + w / 2 - f.pos[0] + 0.005, y + d / 2 - f.pos[1] + 0.005))
        f.blocked_rectangles = tuple(rectangles)

def episode_metadata():
    """Record the calibration and exact layout version alongside each episode."""
    import hashlib
    import json
    from pathlib import Path
    from robocasa.models.scenes.scene_registry import get_layout_path
    path = Path(get_layout_path(100))
    manifest_path = path.with_suffix('.manifest.json')
    manifest = json.loads(manifest_path.read_text())
    assets = path.parents[3]
    paths = [path, manifest_path, path.with_suffix('.walls.json'), assets / 'fixtures/labapt/fridge/model.xml', assets / 'fixtures/labapt/microwave/model.xml']
    return dict(layout_name='LAB_APT', held_out=True, meters_per_unit=1, up_axis='Z', scan_to_layout=manifest['scan_to_layout'], asset_sha256={str(p.relative_to(assets)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}, calibration_status='scan fitted; real-world verification pending')

def adapt_task_placements(env):
    """Keep a task's original goals while locating staging on measured supports."""
    env._labapt_placement_adaptations = []
    if type(env).__name__ != 'DefrostByCategory':
        return
    for cfg in env.object_cfgs:
        placement = cfg.get('placement', {})
        fixture = placement.get('fixture')
        if getattr(fixture, 'name', '') == 'dining_black_table_labapt':
            placement['sample_region_kwargs']['ref'] = fixture
    env._labapt_placement_adaptations.append('DefrostByCategory: separate food and bowl on opposite sides of the dining table')
