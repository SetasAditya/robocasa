"""Native cabinet behavior with placement heights appropriate to LabApt's base cabinets."""
from robocasa.models.fixtures.cabinets import SingleCabinet, HingeCabinet, OpenCabinet

class LabAptCabinetRegions:
    task_selection_z_range = (0.05, 1.5)

    def get_reset_regions(self, env=None, reset_region_names=None, z_range=(0.05, 1.5)):
        return super().get_reset_regions(env=env, reset_region_names=reset_region_names, z_range=z_range)

class LabAptSingleCabinet(LabAptCabinetRegions, SingleCabinet):
    pass

class LabAptHingeCabinet(LabAptCabinetRegions, HingeCabinet):
    pass

class LabAptOpenShelf(OpenCabinet):
    """A thin wall shelf has storage ABOVE its board, not inside the board."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.update_regions({'level0': {'pos': [0, 0, self.size[2] / 2 + 0.175], 'halfsize': [self.size[0] / 2, self.size[1] / 2, 0.175]}}, update_elem=True)

    def get_reset_regions(self, env=None, shelf_level=None, z_range=(0.45, 1.5)):
        z = self.pos[2] + self.size[2] / 2
        if z_range is not None and (not z_range[0] <= z <= z_range[1]):
            return {}
        if shelf_level not in (None, 0):
            return {}
        return {'level0': {'offset': (0, 0, self.size[2] / 2), 'size': tuple(self.size[:2]), 'height': 0.35}}
from robocasa.models.fixtures.counter import Counter
from robocasa.utils.errors import SamplingError
import numpy as np

class LabAptCounter(Counter):
    """Native support surface with occupied appliance footprints removed."""
    blocked_rectangles = ()

    def get_reset_regions(self, env, ref=None, loc='nn', top_size=(0.01, 0.01), **kwargs):
        source = super().get_reset_regions(env, ref=None, loc='any', top_size=top_size, **kwargs)
        result = {}
        for key, region in source.items():
            x, y, z = region['offset']
            w, d = region['size']
            rects = [(x - w / 2, y - d / 2, x + w / 2, y + d / 2)]
            for bx0, by0, bx1, by1 in self.blocked_rectangles:
                pieces = []
                for x0, y0, x1, y1 in rects:
                    ix0, iy0, ix1, iy1 = (max(x0, bx0), max(y0, by0), min(x1, bx1), min(y1, by1))
                    if ix0 >= ix1 or iy0 >= iy1:
                        pieces.append((x0, y0, x1, y1))
                        continue
                    pieces.extend([(x0, y0, ix0, y1), (ix1, y0, x1, y1), (ix0, y0, ix1, iy0), (ix0, iy1, ix1, y1)])
                rects = [r for r in pieces if r[2] - r[0] > 0.045 and r[3] - r[1] > 0.045]
            for i, (x0, y0, x1, y1) in enumerate(rects):
                if ref is not None and loc in ('left', 'right', 'left_right'):
                    f = env.get_fixture(ref)
                    dx = self.pos[0] + (x0 + x1) / 2 - f.pos[0]
                    if loc == 'left' and dx >= 0:
                        continue
                    if loc == 'right' and dx <= 0:
                        continue
                result[f'{key}_free_{i}'] = {'offset': ((x0 + x1) / 2, (y0 + y1) / 2, z), 'size': (x1 - x0, y1 - y0)}
        return result

    def sample_reset_region(self, min_size=None, *args, **kwargs):
        regions = self.get_reset_regions(*args, **kwargs)
        valid = []
        for name, r in regions.items():
            if min_size is not None:
                footprint = np.sort(np.asarray(min_size[:2]))
                surface = np.sort(np.asarray(r['size']))
                if np.any(footprint > surface - 0.042):
                    continue
            valid.append(dict(r, name=name))
        if not valid:
            raise SamplingError(f'No unoccupied LabApt support region fits object {min_size} on {self.name}')
        return self.rng.choice(valid)
from robocasa.models.fixtures.fridge import Fridge

class LabAptFridge(Fridge):
    """Two-door source refrigerator, with shelf ceiling clearance enforced."""

    def get_reset_regions(self, *args, **kwargs):
        regions = super().get_reset_regions(*args, **kwargs)
        for name, region in regions.items():
            cell = self._regions[name]
            region['height'] = float(cell['pz'][2] - cell['p0'][2]) - 0.01
        return regions
