"""Reviewable pilot specification. Candidate assets are not qualified assets."""
TASKS = {
 'surface_delivery': dict(counts=(162,18), envs=('SuiteSurface',), objects=('can','mug','glass_cup','bowl','plate','cereal'), fixtures=('source','target'), stages=('approach','grasp','transport','place')),
 'selective_delivery': dict(counts=(126,14), envs=('SuiteSelective',), objects=('can','mug','cereal'), fixtures=('source','target'), stages=('select','approach','grasp','transport','place')),
 'dishes_to_sink': dict(counts=(126,14), envs=('SuiteSink',), objects=('mug','glass_cup','bowl','plate'), fixtures=('counter','sink'), stages=('approach','grasp','transport','place')),
 'cabinet_articulation': dict(counts=(90,10), envs=('SuiteOpenCabinet','SuiteCloseCabinet'), objects=(), fixtures=('fxtr',), stages=('approach','engage_handle','articulate','release')),
 'drawer_articulation': dict(counts=(90,10), envs=('SuiteOpenDrawer','SuiteCloseDrawer'), objects=(), fixtures=('drawer',), stages=('approach','engage_handle','articulate','release')),
 'cabinet_transfer': dict(counts=(144,16), envs=('SuiteCabinetStore','SuiteCabinetRetrieve'), objects=('can','mug','bowl','cereal'), fixtures=('cab','counter'), stages=('approach','open','grasp','transfer','release','close')),
 'microwave_transfer': dict(counts=(90,10), envs=('SuiteMicrowaveLoad','SuiteMicrowaveUnload'), objects=('mug','bowl','plate'), fixtures=('microwave','counter'), stages=('approach','open','grasp','transfer','release','close')),
 'collect_two_stations': dict(counts=(72,8), envs=('SuiteCollectTwo',), objects=('can','mug','cereal'), fixtures=('source','target'), stages=('approach','grasp','transport','place','approach_second','grasp_second','transport_second','place_second')),
}
SCENES = {'train':dict(layouts=[11,16,24,38],styles=[11,22]),
          'validation':dict(layouts=[45],styles=[33,44]),
          'test':dict(layouts=[8,9,10],styles=[3,8])}
CATEGORIES = ('can','mug','glass_cup','bowl','plate','cereal')
VARIATIONS = dict(base_distance_m=[[.5,.8],[.8,1.5],[1.5,3.]],base_heading_deg=[-90,90],
                  base_band_weights=[.3,.5,.2],clutter_weights=[.3,.4,.3],
                  object_yaw_deg=[0,360],lighting_scale=[.8,1.2],physics='nominal')

for _task, _spec in TASKS.items():
    _spec['success_requirements'] = ['native physical placement or articulation predicate', 'base displacement at least 0.3m']
    if _task in ('cabinet_transfer','microwave_transfer'):
        _spec['success_requirements'] += ['fixture starts closed', 'opening observed before transfer', 'transfer observed', 'fixture ends closed']
    if _task in ('surface_delivery','selective_delivery'):
        _spec['success_requirements'] += ['object contacts target support', 'gripper released and away', 'distractors remain within 4cm of their initial positions']
    if _task == 'collect_two_stations':
        _spec['success_requirements'] += ['both objects contact physical bin floor and lie inside bin', 'gripper away from both objects']
