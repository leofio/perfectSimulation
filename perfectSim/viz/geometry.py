'''
Geometry helpers for the lattice renderer.

Pure numpy: nothing here imports matplotlib, so it is easy to test and reuse
with another plotting backend later.
'''

import numpy as np


def orient(xy: np.ndarray, orientation: str = 'array') -> np.ndarray:
    '''
    Map lattice coordinates to plot coordinates.

    'native': use lat.pos as-is (x = first coord, y = second coord).
    'array' : (i, j) -> (x=j, y=-i), so the picture matches how a state
              array looks when printed / reshaped (row 0 at the top).
    '''
    if orientation == 'native':
        return xy
    if orientation == 'array':
        return np.column_stack([xy[:, 1], -xy[:, 0]])
    raise ValueError(f"orientation must be 'native' or 'array', got {orientation!r}")


def convex_hull(points: np.ndarray) -> np.ndarray:
    '''Andrew's monotone chain. Returns hull vertices in counter-clockwise order.'''
    pts = np.unique(np.round(np.asarray(points, dtype=float), 9), axis=0)  # sorted lexicographically
    if len(pts) <= 2:
        return pts

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in pts[::-1]:
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return np.array(lower[:-1] + upper[:-1])

def padded_hull(points: np.ndarray, radius: float, n_arc: int = 24, ratio: float = 0.4) -> np.ndarray:
    '''
    Boundary of `points`, grown outward by `radius` with rounded corners.
    Automatically uses a concave wrap if the `shapely` library is installed.
    Falls back to a standard convex hull (Minkowski sum) if `shapely` is missing.
    '''
    try:
        from shapely.geometry import MultiPoint
        from shapely import concave_hull
        
        mp = MultiPoint(points)
        hull = concave_hull(mp, ratio=ratio)
        
        padded_shape = hull.buffer(distance=radius, join_style=1)
        
        if hasattr(padded_shape, 'exterior'):
            return np.array(padded_shape.exterior.coords)
            
    except ImportError:
        pass

    ang = np.linspace(0.0, 2 * np.pi, n_arc, endpoint=False)
    circle = radius * np.column_stack([np.cos(ang), np.sin(ang)])
    hull_pts = convex_hull(points)
    cloud = (hull_pts[:, None, :] + circle[None, :, :]).reshape(-1, 2)
    return convex_hull(cloud)

def edge_segments(xy, edges, wrap='stubs', stub=0.35, long_factor=1.5):
    '''
    Turn an edge list into line segments for a LineCollection.

    Returns (segments, owner):
        segments: (S, 2, 2) float array of line segments
        owner:    (S,) int array, owner[s] = id of the edge that segment s belongs to

    Edges much longer than the median edge (wrap-around edges on a torus) would
    slash across the whole picture, so:
        wrap='stubs'    : draw two short stubs pointing off each side (default)
        wrap='hide'     : omit them
        wrap='straight' : draw everything as a straight line (use for irregular graphs)
    '''
    edges = np.asarray(edges)
    if len(edges) == 0:
        return np.empty((0, 2, 2)), np.empty(0, dtype=np.int64)

    p, q = xy[edges[:, 0]], xy[edges[:, 1]]
    d = q - p
    length = np.hypot(d[:, 0], d[:, 1])

    if wrap == 'straight':
        return np.stack([p, q], axis=1), np.arange(len(edges))
    if wrap not in ('stubs', 'hide'):
        raise ValueError(f"wrap must be 'stubs', 'hide' or 'straight', got {wrap!r}")

    long = length > long_factor * np.median(length)
    keep = np.flatnonzero(~long)
    segs = [np.stack([p[keep], q[keep]], axis=1)]
    owner = [keep]

    idx = np.flatnonzero(long)
    if wrap == 'stubs' and idx.size:
        u = d[idx] / length[idx, None]
        segs.append(np.stack([p[idx], p[idx] - stub * u], axis=1))   # leaves p away from q
        segs.append(np.stack([q[idx], q[idx] + stub * u], axis=1))   # leaves q away from p
        owner += [idx, idx]

    return np.concatenate(segs), np.concatenate(owner)
