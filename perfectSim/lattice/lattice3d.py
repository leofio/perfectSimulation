'''
3D lattices. Builds on perfectSim.lattice (Lattice, make_lattice, FREE), so
boundary_values(), the models and the visualiser all work unchanged.

Coordinates are (i, j, k) index triples stored in lat.pos, shape (n_total, 3).
Site id = (i * M + j) * N + k.
'''

import itertools
from dataclasses import replace
import numpy as np
from perfectSim.lattice.lattice import FREE, Lattice, make_lattice, complete

CUBIC = [(-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)]


def _grid3(shape, offsets, is_bipartite, colour=None, periodic=False, ghosts=False) -> Lattice:
    '''3D grid with optional periodic wrap or a one-layer ghost boundary.'''
    L, M, N = shape
    n = L * M * N
    ghost = {}

    def nid(i, j, k):
        if periodic:
            return ((i % L) * M + (j % M)) * N + (k % N)
        if 0 <= i < L and 0 <= j < M and 0 <= k < N:
            return (i * M + j) * N + k
        if not ghosts:
            return FREE
        if (i, j, k) not in ghost:
            ghost[(i, j, k)] = n + len(ghost)
        return ghost[(i, j, k)]

    sites = list(itertools.product(range(L), range(M), range(N)))
    nbr = np.array([[nid(i + di, j + dj, k + dk) for di, dj, dk in offsets]
                    for i, j, k in sites], dtype=np.int32)
    ghost_list = sorted(ghost, key=ghost.get)
    pos = np.array(sites + ghost_list, dtype=np.float32)
    ghost_pos = np.array(ghost_list, dtype=np.float32) if ghost_list else None
    return make_lattice(n, nbr, is_bipartite, colour=colour, n_boundary=len(ghost),
                        ghost_pos=ghost_pos, pos=pos)


def _colour(L, M, N):
    return np.array([(i + j + k) % 2 for i in range(L) for j in range(M) for k in range(N)],
                    dtype=np.int8)


def grid3d(L, M=None, N=None, ghosts=False) -> Lattice:
    '''L x M x N cubic grid (bipartite). ghosts=True adds a one-site-thick boundary shell
    on the six faces (no edge/corner ghosts are needed: only face neighbours exist).'''
    M = L if M is None else M
    N = M if N is None else N
    return _grid3((L, M, N), CUBIC, True, colour=_colour(L, M, N), ghosts=ghosts)


def torus3d(L, M=None, N=None) -> Lattice:
    '''L x M x N periodic cubic lattice (bipartite iff all sides are even; sides must be >= 2).'''
    M = L if M is None else M
    N = M if N is None else N
    even = L % 2 == 0 and M % 2 == 0 and N % 2 == 0
    return _grid3((L, M, N), CUBIC, even, colour=_colour(L, M, N) if even else None, periodic=True)


def complete_sphere(n) -> Lattice:
    '''Complete graph K_n with sites spread over a unit sphere (Fibonacci layout), for 3D plots.'''
    lat = complete(n)
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n)
    theta = np.pi * (1 + np.sqrt(5)) * i
    pos = np.stack([np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)],
                   axis=1).astype(np.float32)
    return replace(lat, pos=pos)
