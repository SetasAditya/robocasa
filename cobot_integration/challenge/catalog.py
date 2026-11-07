"""Versioned challenge recipes. Importable without MuJoCo or RoboCasa assets."""
from copy import deepcopy

VERSION = 'labapt-challenge-v1'
DIFFICULTIES = ('easy', 'medium', 'hard')
MODES = ('workflow', 'manipulation')
PROFILES = {
 'OccludedMugRetrieval': ('constrained', 'Retrieve the mug from behind the protected objects and release it on the counter.', 'cabinet', 'mug', 'occlusion', [1, 2, 3]),
 'CrowdedRackMugPlacement': ('constrained', 'Place the mug upright in the marked dishwasher rack region without moving the protected dishes.', 'rack', 'mug', 'clearance_m', [.08, .05, .025]),
 'ShallowDrawerRetrieval': ('constrained', 'Retrieve the shallow utensil from the drawer and release it on the counter.', 'drawer', 'wooden_spoon', 'drawer_open_fraction', [1., .75, .55]),
 'CabinetBowlStacking': ('constrained', 'Build a stable two-bowl stack in the cabinet, with the target bowl upright on the inverted support bowl; do not nest them.', 'stack', 'bowl', 'alignment_tolerance_m', [.04, .025, .015]),
 'RestrictedDoorRetrieval': ('constrained', 'Open the partially open cabinet further, retrieve the mug, and release it on the counter.', 'door', 'mug', 'door_open_fraction', [.65, .45, .25]),
 'ImperfectGraspRecovery': ('constrained', 'Recover the imperfect grasp and release the mug upright on the marked counter region.', 'recovery', 'mug', 'grasp_offset_m', [.005, .015, .025]),
 'PlateRackInsertion': ('contact', 'Insert the plate upright into the marked dishwasher rack slot and release it stably.', 'plate', 'plate', 'clearance_m', [.03, .02, .01]),
 'CabinetBowlNesting': ('contact', 'Seat the target bowl inside the support bowl in the cabinet, then release it.', 'nest', 'bowl', 'alignment_tolerance_m', [.035, .02, .012]),
 'GuidedTrayInsertion': ('contact', 'Slide the tray into the blue countertop guides, seat it against the back stop, then release it.', 'tray', 'oven_tray', 'clearance_m', [.03, .02, .01]),
 'SpongeContactCoverage': ('contact', 'Grasp the sponge and wipe at least 90 percent of the marked patch through physical contact, then release it on the counter.', 'wipe', 'sponge', 'patch_width_m', [.12, .18, .24]),
}


def recipe(profile, difficulty='easy', mode='workflow'):
    if profile not in PROFILES or difficulty not in DIFFICULTIES or mode not in MODES:
        raise ValueError('Unknown profile, difficulty or initialization mode')
    family, instruction, kind, group, axis, values = PROFILES[profile]
    return dict(version=VERSION, profile=profile, env_name='LabApt'+profile,
                family=family, instruction=instruction, kind=kind, obj_group=group,
                difficulty=difficulty, mode=mode, primary_axis=axis,
                parameters={axis: values[DIFFICULTIES.index(difficulty)]},
                layout=100, style=1, control_hz=20, horizon=1800,
                stable_seconds=1., linear_speed_limit_m_s=.025,
                angular_speed_limit_rad_s=.15, protected_displacement_m=.025,
                protected_tilt_rad=.15, max_penetration_m=.005,
                status='experimental', qualification='requires_frozen_scene_and_witness',
                physical_validation=False)


def plan(profile, difficulty='easy', mode='workflow', split='practice', count=None):
    if split not in ('development', 'practice', 'formal'):
        raise ValueError('Unknown split')
    count = (50 if split == 'formal' else 20) if count is None else count
    if type(count) is not int or count < 1:
        raise ValueError('count must be a positive integer')
    r = recipe(profile, difficulty, mode)
    # Different cells and splits have disjoint seeds; methods share each cell.
    cell = list(PROFILES).index(profile)*6 + DIFFICULTIES.index(difficulty)*2 + MODES.index(mode)
    if count > 10000:
        raise ValueError('count exceeds reserved seed interval')
    start = {'development': 1000000, 'practice': 2000000, 'formal': 3000000}[split] + cell*10000
    return dict(recipe=deepcopy(r), split=split,
                scenes=[dict(id=f'{profile}/{difficulty}/{mode}/{split}/{i:03d}', seed=start+i,
                             status='pending_preparation') for i in range(count)])
