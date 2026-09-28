'''
Palettes: map raw state values (ints) to colours.

The renderer never looks at the model. It only sees integer values, and a
Palette says what colour each one is. That is what lets one renderer serve
Ising (+-1), hard-core (0/1), Potts (0..q-1) and bounding chains (extra
'unknown' value).
'''

from typing import Mapping
import numpy as np
from matplotlib import colormaps
from matplotlib.colors import to_rgba


class Palette:
    '''
    colours:  {value: matplotlib colour}
    fallback: colour for any value not in the mapping (including FREE boundary sites)
    '''
    def __init__(self, colours: Mapping[int, object], fallback='0.6'):
        keys = sorted(int(k) for k in colours)
        self.colours = {int(k): v for k, v in colours.items()}
        self.lo, self.hi = keys[0], keys[-1]
        self.fallback = np.array(to_rgba(fallback))
        self._lut = np.tile(self.fallback, (self.hi - self.lo + 1, 1))
        for k, c in self.colours.items():
            self._lut[k - self.lo] = to_rgba(c)

    def rgba(self, values) -> np.ndarray:
        '''(n,) ints -> (n, 4) RGBA floats. Vectorised; out-of-range values get the fallback.'''
        idx = np.asarray(values).astype(np.int64) - self.lo
        ok = (idx >= 0) & (idx < len(self._lut))
        out = np.tile(self.fallback, (len(idx), 1))
        out[ok] = self._lut[idx[ok]]
        return out


UNKNOWN = '#d0d0d0'

ISING = Palette({-1: '#3b6fd4', 1: '#e8743b', 2: UNKNOWN})
HARDCORE = Palette({0: 'white', 1: '#2a9d8f', 2: UNKNOWN})


def categorical_palette(k: int) -> Palette:
    '''k distinguishable colours for values 0..k-1 (Potts). tab10 up to 10, hue wheel above.'''
    if k <= 10:
        cmap = colormaps['tab10']
        return Palette({i: cmap(i) for i in range(k)})
    cmap = colormaps['hsv']
    return Palette({i: cmap(i / k) for i in range(k)})


def default_palette(model=None, values=None) -> Palette:
    '''
    Pick a sensible palette. Priority: model type, then a guess from the values.
    (Guessing: any negative value -> Ising; otherwise categorical with max+1 colours.)
    '''
    name = type(model).__name__ if model is not None else ''
    if 'Ising' in name:
        return ISING
    if 'HardCore' in name:
        return HARDCORE
    if values is not None and np.size(values):
        v = np.asarray(values)
        if v.min() < 0:
            return ISING
        return categorical_palette(max(int(v.max()) + 1, 2))
    return ISING
