'''
Ising model. So far only set up for LxL torus lattice.
'''

import numpy as np
from src.lattice import Lattice
from src.baseModel import BaseModel

class Ising(BaseModel):
    def __init__(self, lattice: Lattice, beta, h = 0.0):
        self.lattice = lattice
        self.beta = beta
        self.h = h
        self.table = np.array([
            1 / (1 + np.exp(-2 * (beta * S + h))) for S in (-4, -2, 0, 2, 4)
        ])

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
    
    def equal(self, x, y):
        '''Check two states are equal'''
        return np.array_equal(x, y)

    def leq(self, x, y):
        '''Check partial order'''
        return bool(np.all(x <= y))

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
        states = states.copy()
        sites, unifs = r
        batch_size, k = sites.shape

        batch_indices = np.arange(batch_size)

        for step in range(k):
            v = sites[:, step]
            u = unifs[:, step]

            nbrs = self.lattice.nbr[v]
            neighbour_states = states[batch_indices[:, None], nbrs]

            S = neighbour_states.sum(axis=1)
            p = self.table[(S + 4) // 2]

            states[batch_indices, v] = np.where(u <= p, 1, -1)

        return states
    