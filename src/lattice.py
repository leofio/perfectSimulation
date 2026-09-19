'''
Lattices
'''

from dataclasses import dataclass
from typing import Optional
import numpy as np

@dataclass(frozen=True)
class Lattice:
    n_sites: int
    nbr: np.ndarray # (n_sites, 4) nbr[v] contains the four neighbours of v

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
    return Lattice(n_sites=n, nbr=nbr)