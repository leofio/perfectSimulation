'''
Geometry helpers for the 3D lattice renderer.

Pure numpy (scipy is used opportunistically for the boundary hull): nothing
here imports matplotlib.
'''

import itertools
import numpy as np


def as_3d(pos: np.ndarray) -> np.ndarray:
    '''(n, 2) or (n, 3) coordinates -> (n, 3) float array (2D lattices get z = 0).'''
    pos = np.asarray(pos, dtype=float)
    if pos.ndim != 2 or pos.shape[1] not in (2, 3):
        raise NotImplementedError(f'expected pos of shape (n, 2) or (n, 3), got {pos.shape}')
    if pos.shape[1] == 2:
        pos = np.column_stack([pos, np.zeros(len(pos))])
    return pos


def orient(pos: np.ndarray, orientation: str = 'array') -> np.ndarray:
    '''
    Map lattice coordinates to plot coordinates (always returns (n, 3)).

    'native': use lat.pos as-is (x, y, z = first, second, third coord).
    'array' : the picture matches how a state array looks when indexed a[i, j, k]:
                3D lattices: (i, j, k) -> (x=k, y=j, z=-i)  (row 0 on top, right-handed)
                2D lattices: (i, j)    -> (x=j, y=-i, z=0)  (identical to the 2D package)
    '''
    is_2d = np.asarray(pos).shape[1] == 2
    xyz = as_3d(pos)
    if orientation == 'native':
        return xyz
    if orientation == 'array':
        if is_2d:
            return np.column_stack([xyz[:, 1], -xyz[:, 0], xyz[:, 2]])
        return np.column_stack([xyz[:, 2], xyz[:, 1], -xyz[:, 0]])
    raise ValueError(f"orientation must be 'native' or 'array', got {orientation!r}")


def _box_faces(lo, hi):
    '''12 triangles covering the surface of an axis-aligned box.'''
    faces = []
    for a in range(3):
        b, c = [i for i in range(3) if i != a]
        for side in (lo[a], hi[a]):
            def pt(u, v):
                p = np.empty(3)
                p[a], p[b], p[c] = side, u, v
                return p
            q = [pt(lo[b], lo[c]), pt(hi[b], lo[c]), pt(hi[b], hi[c]), pt(lo[b], hi[c])]
            faces += [[q[0], q[1], q[2]], [q[0], q[2], q[3]]]
    return np.array(faces)


def padded_hull_faces(points: np.ndarray, radius: float) -> np.ndarray:
    '''
    Triangles (F, 3, 3) of the convex hull of `points` grown outward by `radius`
    (Minkowski sum with a cube). Uses scipy's ConvexHull if available; otherwise,
    or if the hull is degenerate, falls back to the padded bounding box.
    '''
    pts = np.asarray(points, dtype=float)
    signs = np.array(list(itertools.product((-1.0, 1.0), repeat=3))) * radius
    cloud = (pts[:, None, :] + signs[None, :, :]).reshape(-1, 3)
    try:
        from scipy.spatial import ConvexHull
        return cloud[ConvexHull(cloud).simplices]
    except Exception:
        return _box_faces(cloud.min(axis=0), cloud.max(axis=0))


def edge_segments(xyz, edges, wrap='stubs', stub=0.35, long_factor=1.5):
    '''
    Turn an edge list into 3D line segments for a Line3DCollection.

    Returns (segments, owner):
        segments: (S, 2, 3) float array
        owner:    (S,) int array, owner[s] = id of the edge segment s belongs to

    Edges much longer than the median edge (torus wrap-arounds) are handled by:
        wrap='stubs'    : two short stubs pointing off each side (default)
        wrap='hide'     : omitted
        wrap='straight' : drawn as straight lines (irregular graphs, K_n, trees)
    '''
    edges = np.asarray(edges)
    if len(edges) == 0:
        return np.empty((0, 2, 3)), np.empty(0, dtype=np.int64)

    p, q = xyz[edges[:, 0]], xyz[edges[:, 1]]
    d = q - p
    length = np.linalg.norm(d, axis=1)

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
        segs.append(np.stack([p[idx], p[idx] - stub * u], axis=1))
        segs.append(np.stack([q[idx], q[idx] + stub * u], axis=1))
        owner += [idx, idx]

    return np.concatenate(segs), np.concatenate(owner)
