"""Small geometry predicates shared by the simulator and shortcut regressions."""
import math


def released_rest(*, in_destination, released, linear_speed, angular_speed,
                  linear_limit=.025, angular_limit=.15):
    return bool(in_destination and released and math.isfinite(linear_speed)
                and math.isfinite(angular_speed) and linear_speed < linear_limit
                and angular_speed < angular_limit)


def rack_seated(*, delta, half_extents, orientation_alignment, rack_contact):
    return bool(rack_contact and orientation_alignment > math.cos(.15)
                and all(math.isfinite(v) and abs(v) <= bound for v,bound in zip(delta, half_extents))
                and len(delta)==len(half_extents)==3 and all(v>0 for v in half_extents))


def bowl_seated(*, height, height_interval, xy_distance, tolerance,
                target_upright, support_upright, contact, both_inside, support_quiet,
                support_released):
    return bool(height_interval and height_interval[0] <= height <= height_interval[1]
                and xy_distance <= tolerance and target_upright and support_upright
                and contact and both_inside and support_quiet and support_released)


def validate_geometry(kind, geometry):
    """Reject malformed operator geometry; physical fit still requires review."""
    if kind in ('plate','rack'):
        slot=geometry.get('slot')
        if not slot or len(slot.get('center',[]))!=3 or len(slot.get('half_extents',[]))!=3:
            raise ValueError('Rack geometry needs center and half_extents vectors')
        if not all(math.isfinite(v) for v in slot['center']+slot['half_extents']) or min(slot['half_extents'])<=0:
            raise ValueError('Invalid rack bounds')
        if slot.get('plate_normal_axis',0) not in (0,1,2):
            raise ValueError('Invalid plate normal axis')
    elif kind in ('stack','nest'):
        interval=geometry.get('seated_height_interval_m')
        if not interval or len(interval)!=2 or not all(math.isfinite(v) for v in interval) or not 0<interval[0]<interval[1]:
            raise ValueError('Bowl geometry needs a positive ordered seated height interval')
        if geometry.get('relationship') != kind:
            raise ValueError('Stack and nest require separately reviewed geometry')
    elif kind=='recovery':
        nominal=geometry.get('nominal_grasp_tool_xyz')
        if not nominal or len(nominal)!=3 or not all(math.isfinite(v) for v in nominal):
            raise ValueError('Recovery requires a finite nominal grasp vector')


def tray_guides(size, clearance):
    """Rails fit the actual model footprint; the back stop seats its center at zero."""
    width, depth = float(size[0]), float(size[1])
    if not all(math.isfinite(v) and v>0 for v in (width,depth,clearance)):
        raise ValueError('Invalid tray footprint or clearance')
    halfwidth = width/2+clearance/2
    rail_length = depth/2+.08
    return [([-halfwidth-.005,0.,.025],[.005,rail_length,.025]),
            ([halfwidth+.005,0.,.025],[.005,rail_length,.025]),
            ([0.,depth/2+.005,.025],[halfwidth+.01,.005,.025])]
