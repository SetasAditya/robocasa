"""Headless LabApt geometry, articulation, scene rendering, and reset checks."""
import argparse, json, os
from pathlib import Path
os.environ.setdefault('MUJOCO_GL', 'egl')
import numpy as np

def scene(out, render):
    import mujoco
    from robocasa.models.scenes.kitchen_arena import KitchenArena
    from robocasa.models.scenes.scene_registry import get_layout_path, unpack_layout_ids, LayoutType
    from robocasa.models.fixtures.fixture import FixtureType
    from robocasa.models.fixtures.fixture_utils import fixture_is_type

    def matches(f, t):
        try:
            return fixture_is_type(f, t)
        except ValueError:
            return False
    arena = KitchenArena(100, 1, rng=np.random.default_rng(0))
    manifest = json.loads(Path(get_layout_path(100)).with_suffix('.manifest.json').read_text())
    expected = {r['name'] + '_labapt': r for r in manifest['fixtures']}
    data = {'fixtures': {}, 'geometry_failures': [], 'articulation': [], 'task_rollouts': 'not tested'}
    for name, f in arena.fixtures.items():
        arena.merge_assets(f)
        arena.worldbody.append(f.get_obj())
        record = {'class': type(f).__name__, 'position': np.asarray(f.pos).tolist(), 'size': np.asarray(f.size).tolist(), 'types': [t.name for t in FixtureType if matches(f, t)]}
        if name in expected:
            ref = expected[name]
            record['size_error_m'] = float(np.max(np.abs(np.asarray(f.size) - ref['dimensions_m'])))
            actual = np.asarray(f.pos) + np.asarray(getattr(f, 'origin_offset', [0, 0, 0]))
            record['position_reference'] = 'bounding-box center'
            if name == 'fridge_labapt':
                actual = np.asarray(f.pos)
                record['position_reference'] = 'source USD root translated by half height'
            elif name == 'sink_labapt':
                actual = actual + np.array([0, 0, f.size[2] / 2])
                record['position_reference'] = 'countertop installation anchor'
            record['position_error_m'] = float(np.max(np.abs(actual - ref['position_m'])))
            if record['size_error_m'] > 0.001:
                data['geometry_failures'].append(name + ': size')
            if record['position_error_m'] > 0.001:
                data['geometry_failures'].append(name + ': position')
        data['fixtures'][name] = record
    assert 100 not in unpack_layout_ids(LayoutType.TRAIN)
    assert unpack_layout_ids('LAB_APT') == [100]
    import xml.etree.ElementTree as ET
    compiler = arena.root.find('compiler')
    if compiler is None:
        compiler = ET.SubElement(arena.root, 'compiler')
    compiler.set('angle', 'radian')
    xml = arena.get_xml()
    (out / 'scene.xml').write_text(xml)
    model = mujoco.MjModel.from_xml_string(xml)
    state = mujoco.MjData(model)
    mujoco.mj_forward(model, state)
    data.update(nbody=model.nbody, ngeom=model.ngeom, njnt=model.njnt)
    q0 = state.qpos.copy()
    for j in range(model.njnt):
        if model.jnt_type[j] not in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
            continue
        lo, hi = model.jnt_range[j]
        adr = model.jnt_qposadr[j]
        for v in np.linspace(lo, hi, 11):
            state.qpos[:] = q0
            state.qpos[adr] = v
            mujoco.mj_forward(model, state)
            assert np.isfinite(state.geom_xpos).all()
        data['articulation'].append({'joint': mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j), 'range': [float(lo), float(hi)], 'kinematic_sweep': 'pass', 'physical_robot_manipulation': 'not tested'})
    state.qpos[:] = q0
    mujoco.mj_forward(model, state)
    if render:
        from PIL import Image
        model.vis.global_.offwidth = 1280
        model.vis.global_.offheight = 720
        renderer = mujoco.Renderer(model, height=720, width=1280)
        for name, look, distance, azimuth, elevation in [('overview', [-0.1, -1.6, 0.75], 6.2, 90, -35), ('kitchen', [-0.25, -0.05, 0.9], 3.7, 90, -15), ('table', [-0.15, -3.5, 0.5], 3.3, 45, -35)]:
            cam = mujoco.MjvCamera()
            cam.lookat[:] = look
            cam.distance = distance
            cam.azimuth = azimuth
            cam.elevation = elevation
            option = mujoco.MjvOption()
            option.geomgroup[0] = 0
            renderer.update_scene(state, camera=cam, scene_option=option)
            Image.fromarray(renderer.render()).save(out / (name + '.png'))
        renderer.close()
    (out / 'scene_report.json').write_text(json.dumps(data, indent=2) + '\n')
    print(json.dumps({'scene': 'compiled', 'joints': model.njnt, 'geometry_failures': data['geometry_failures']}), flush=True)
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--render', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    scene(args.output, args.render)
