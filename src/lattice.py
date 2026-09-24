'''
Lattices
'''

from dataclasses import dataclass
from typing import Optional, Union
import numpy as np

FREE = -128

@dataclass(frozen=True, eq=False)
class Lattice:
    n_sites: int
    nbr: np.ndarray # (n_sites, max_degree) nbr[v] contains the neighbours of v
    n_boundary: int = 0
    # The Lattice holds ghost_states that represent boudnary nodes, 
    # it is left to the model to assign values to the ghost nodes
    # to impose boundary conditions.
    ghost_pos: Optional[np.ndarray] = None # (n_boundary, 2) grid coords of boundary sites

    @property
    def max_degree(self) -> int: return self.nbr.shape[1]

    @property
    def null(self) -> int: return self.n_sites + self.n_boundary

def make_lattice(n_sites, nbr, n_boundary=0, ghost_pos=None) -> Lattice:
    nbr = np.asarray(nbr, dtype=np.int32)
    null = n_sites + n_boundary
    nbr = np.where(nbr == FREE, null, nbr).astype(np.int32)

    assert nbr.shape[0] == n_sites
    assert ((nbr >= 0) & (nbr <= null)).all(), 'neighbour id out of range'
    assert not (nbr == np.arange(n_sites)[:, None]).any(), 'self-loop'

    real = [(v, u) for v in range(n_sites) for u in nbr[v] if u < n_sites]
    assert sorted(real) == sorted((u, v) for v, u in real), 'real-site adjacency is not symmetric'

    return Lattice(
        n_sites=n_sites, 
        nbr=nbr, 
        n_boundary=n_boundary, 
        ghost_pos=ghost_pos
    )

def _grid(L, M, offsets, periodic=False, ghosts=False):
    '''
    Create a grid lattice with option to include a boundary.
    '''
    n = L * M
    ghost = {}

    def nid(i, j):
        if periodic:
            return (i % L) * M + (j % M)
        if 0 <= i < L and 0 <= j < M:
            return i * M + j
        if not ghosts:
            return FREE
        if (i, j) not in ghost:
            ghost[(i, j)] = n + len(ghost)
        return ghost[(i, j)]

    nbr = np.array([[nid(i + di, j + dj) for di, dj in offsets]
                    for i in range(L) for j in range(M)], dtype=np.int32)
    pos = np.array(sorted(ghost, key=ghost.get), dtype=np.int32).reshape(-1, 2)
    return make_lattice(n, nbr, n_boundary=len(ghost), ghost_pos=pos)

SQUARE = [(-1, 0), (1, 0), (0, -1), (0, 1)]
TRIANGULAR = SQUARE + [(-1, 1), (1, -1)]

def torus(L, M=None) -> Lattice:
    '''Create an LxM torus lattice'''
    M = L if M is None else M
    return _grid(L, M, SQUARE, periodic=True)

def grid(L, M=None, ghosts=False):
    '''LxM square gird.'''
    M = L if M is None else M
    return _grid(L, M, SQUARE, ghosts=ghosts)

def triangular(L, M=None, ghosts=False):
    '''LxM triangular grid.'''
    M = L if M is None else M
    return _grid(L, M, TRIANGULAR, ghosts=ghosts)

def boundary_values(lat: Lattice, spec: Union[int, callable]):
    '''
    spec: int, or callable(gi, gj) -> array of values (FREE entries skipped).
    Returns {ghost_id: value} for passing to a model.
    '''
    if lat.n_boundary == 0: return {}
    gi, gj = lat.ghost_pos.T
    vals = np.broadcast_to(spec(gi, gj) if callable(spec) else spec, gi.shape)
    return {lat.n_sites + g: int(v) for g, v in enumerate(vals) if v != FREE}