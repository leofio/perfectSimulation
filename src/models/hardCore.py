'''
Hard-Core gas model.
'''

import numpy as np
from src.lattice import Lattice
from src.models.baseModel import MonotoneModel

class HardCoreBipartite(MonotoneModel):
    def __init__(self, lattice: Lattice, activity: float, boundary = None):
        super().__init__(lattice)

        assert activity > 0, 'Hard-core model requires activity > 0'
        self.activity = activity
        self.p = activity / (1.0 + activity)

        assert lattice.is_bipartite, 'HardCoreBipartite model requires a bipartite graph'

        boundary = boundary or {}
        assert all(s in (0, 1) for s in boundary.values()), 'Hard-core boudnary must be 0s or 1s'
        assert all(self.n <= g < lattice.null for g in boundary), 'Boundary keys must be ghost ids'
        self.n_ext = lattice.n_boundary + 1
        self.bvals = np.zeros(self.n_ext, dtype=np.int8)
        for g, s in boundary.items():
            self.bvals[g - self.n] = s

    @property
    def n(self): return self.lattice.n_sites

    @property
    def bottom(self):
        return self.lattice.colour

    @property
    def top(self):
        return 1 - self.lattice.colour

    def leq(self, x: np.ndarray, y: np.ndarray):
        '''
        Check partial order
        For even sites (colour 0): x_i <= y_i
        For odd sites (colour 1): y_i <= x_i
        '''
        return bool(np.all(np.where(self.lattice.colour == 0, x <= y, x>= y)))

    def randomness(self, depth, keys, k):
        '''
        Get the randomness for k steps over a batch of keys.
        keys: (batch_size,) array of seed keys
        Returns sites and unifs of shape (batch_size, k)
        '''
        batch_size = len(keys)
        sites = np.empty((batch_size, k), dtype=np.int32)
        unifs = np.empty((batch_size, k), dtype=np.float64)

        for i, key in enumerate(keys):
            rng = np.random.Generator(np.random.Philox(seed=key, counter=depth))
            sites[i] = rng.integers(0, self.n, k)
            unifs[i] = rng.random(k)

        return sites, unifs

    def apply(self, states, r):
        '''
        One batch of k state updates.
        states: (batch_size, n)
        r: Tuple of sites, unifs each of shape (batch_size, k)
        '''
        sites, unifs = r
        batch_size, k = sites.shape

        batch_indices = np.arange(batch_size)

        padded = np.empty((batch_size, self.n + self.n_ext), dtype=np.int8)
        padded[:, :self.n] = states
        padded[:, self.n:] = self.bvals

        for step in range(k):
            v = sites[:, step]
            u = unifs[:, step]

            nbrs = self.lattice.nbr[v]
            neighbour_states = padded[batch_indices[:, None], nbrs]

            S = neighbour_states.sum(axis=1, dtype=np.int32)
            padded[batch_indices, v] = np.where((S == 0) & (u < self.p), 1, 0)

        return padded[:, :self.n]