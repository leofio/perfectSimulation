'''
Lattices
'''

from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class Lattice:
    n_sites: int
    nbr: np.ndarray # (n_sites, 4) nbr[v] contains the four neighbours of v