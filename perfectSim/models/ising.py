'''
Ising model.
'''

import numpy as np
from perfectSim.lattice import Lattice
from perfectSim.models.baseModel import MonotoneModel, BoundingModel

class Ising(MonotoneModel):
    def __init__(self, lattice: Lattice, beta, h = 0.0, boundary=None):
        super().__init__(lattice)

        assert beta >=0, 'monotone Ising model requites beta>=0'
        self.beta = beta
        self.h = h
        self.table = np.array([
            1 / (1 + np.exp(-2 * (beta * S + h))) for S in range(-lattice.max_degree, lattice.max_degree + 1)
        ])

        boundary = boundary or {}
        assert all(s in (-1, 1) for s in boundary.values()), "Ising boundary must be +-1"
        assert all(self.n <= g < lattice.null for g in boundary), 'Boundary keys must be ghost ids'
        self.n_ext = lattice.n_boundary + 1
        self.bvals = np.zeros(self.n_ext, dtype=np.int8)
        for g, s in boundary.items():
            self.bvals[g - self.n] = s

    @property
    def n(self):
        return self.lattice.n_sites

    @property
    def bottom(self):
        '''Lowest state in partial order'''
        return -np.ones(self.n, dtype = np.int8)

    @property
    def top(self):
        '''Highest state in partial order'''
        return np.ones(self.n, dtype = np.int8)

    def leq(self, x, y):
        '''Check partial order'''
        return bool(np.all(x <= y))

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
            p = self.table[S + self.lattice.max_degree]
            padded[batch_indices, v] = np.where(u <= p, 1, -1)

        return padded[:, :self.n]

class IsingAntiFerromagnetic(BoundingModel):
    def __init__(self, lattice, beta, h = 0.0, boundary=None):
        super().__init__(lattice)

        assert beta < 0, 'Antiferromagnetic Ising model requires beta < 0'
        self.beta = beta
        self.h = h

        self.table = np.array([
            1 / (1 + np.exp(-2 * (beta * S + h))) for S in range(-lattice.max_degree, lattice.max_degree + 1)
        ])

        boundary = boundary or {}
        assert all(s in (-1, 1) for s in boundary.values()), "Ising boundary must be +-1"
        assert all(self.n <= g < lattice.null for g in boundary), 'Boundary keys must be ghost ids'
        self.n_ext = lattice.n_boundary + 1
        self.bvals = np.zeros(self.n_ext, dtype=np.int8)
        for g, s in boundary.items():
            self.bvals[g - self.n] = s

    @property
    def n(self): return self.lattice.n_sites

    @property
    def unknown(self):
        return 2

    def apply(self, states, r):
        '''
        One batch of k state updates for the bounding chain.
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

            n_unknown = (neighbour_states == self.unknown).sum(axis=1, dtype=np.int32)

            known_only = np.where(neighbour_states == self.unknown, 0, neighbour_states)
            S_known = known_only.sum(axis=1, dtype=np.int32)

            S_min = S_known - n_unknown
            S_max = S_known + n_unknown

            p_max = self.table[S_min + self.lattice.max_degree]
            p_min = self.table[S_max + self.lattice.max_degree]

            new_state = np.where(u <= p_min, 1, np.where(u > p_max, -1, self.unknown))

            padded[batch_indices, v] = new_state

        return padded[:, :self.n]
