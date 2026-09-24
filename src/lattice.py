'''
Lattices
'''

from dataclasses import dataclass, replace
from typing import Optional, Union
import numpy as np

FREE = -128

@dataclass(frozen=True, eq=False)
class Lattice:
    n_sites: int
    nbr: np.ndarray # (n_sites, max_degree) nbr[v] contains the neighbours of v
    shape: tuple
    boundary: Optional[np.ndarray] = None # boundary[g] is the state of ghost site with id g

    @property
    def max_degree(self) -> int:
        return self.nbr.shape[1]

    @property
    def n_ghost_lat(self) -> int:
        L, M = self.shape
        return (L+2)*(M+2)

def _nid(i, j, L, M):
    if 0 <= i < L and 0 <= j < M:
        return i * M + j
    return L * M + (i + 1) * (M + 2) + (j + 1)

def apply_boundary(lat: Lattice, spec: Union[int, callable]) -> Lattice:
    '''
    spec: int               -> same state for the entire boundary
          callable(gi, gj)  -> array of states, one per boundary site
    FREE entries are ignored
    '''
    L, M = lat.shape
    g = np.arange(lat.n_ghost_lat)
    gi, gj = g // (M + 2) - 1, g % (M + 2) - 1

    if lat.boundary is not None:
        state = lat.boundary.copy()
    else:
        state = np.full(lat.n_ghost_lat, FREE, np.int8)

    if callable(spec):
        new_state = spec(gi, gj)
        state = np.where(new_state != FREE, new_state, state)
    else:
        state[:] = spec
    return replace(lat, boundary=state)

def torus(L: int, M: Optional[int] = None) -> Lattice:
    '''L x M periodic lattice. Sites are indexed by v = i*M + j.'''
    M = L if M is None else M
    n = L * M
    nbr = np.empty((n, 4), dtype = np.int32)
    for i in range(L):
        for j in range(M):
            v = i * M + j
            nbr[v] = [
                ((i-1) % L) * M + j,
                ((i+1) % L) * M + j,
                i * M + (j - 1) % M,
                i * M + (j + 1) % M
            ]
    return Lattice(n_sites=n, nbr=nbr, shape=(L, M))

def free_grid(L: int, M: Optional[int] = None) -> Lattice:
    '''
    L x M free grid lattice. Sites indexed by v = i*M + j.
    Index n represents a free boundary.
    '''
    M = L if M is None else M
    n = L * M

    nbr = np.full((n, 4), n, dtype=np.int32)

    for i in range(L):
        for j in range(M):
            nbr[i*M + j] = [_nid(i-1, j, L, M), _nid(i+1, j, L, M),
                            _nid(i, j-1, L, M), _nid(i, j+1, L, M)]
    return Lattice(n_sites=n, nbr=nbr, shape=(L, M))

def free_triangular(L: int, M: Optional[int] = None) -> Lattice:
    '''
    L x M triangular lattice with free boundaries. Sites indexed by v = i*M + j. 
    Index n represents a free boundary.
    '''
    M = L if M is None else M
    n = L * M

    nbr = np.full((n, 6), n, dtype=np.int32)

    for i in range(L):
        for j in range(M):
            nbr[i*M + j] = [_nid(i-1, j, L, M), _nid(i+1, j, L, M),
                            _nid(i, j-1, L, M), _nid(i, j+1, L, M),
                            _nid(i-1, j+1, L, M), _nid(i+1, j-1, L, M)]

    return Lattice(n, nbr, (L, M))
