'''
Ising model. So far only set up for LxL torus lattice.
'''

import numpy as np
from lattice.lattice import Lattice

class Ising:
    def __init__(self, lattice: Lattice, beta, h = 0.0):
        self.lattice = lattice
        self.n = lattice.n_sites
        self.beta = beta
        self.h = h
        self.table = np.array([
            1 / (1 + np.exp(-2 * (beta * S + h))) for S in (-4, -2, 0, 2, 4)
        ])

    def bottom(self):
        '''Lowest state in partial order'''
        return -np.ones((self.n,))
    
    def top(self):
        '''Highest state in partial order'''
        return np.ones((self.n,))

    def equal(self, x, y):
        '''Check two states are equal'''
        return np.array_equal(x, y)

    def randomness(self, t, key, k):
        '''Get the randomness for k steps'''
        rng = np.random.Generator(np.random.Philox(key = key, counter = t))
        return rng.integers(0, self.n, k), rng.random(k)

    def apply(self, state, r):
        '''
        One batch of k state updates, deterministic in (state, r)
        k is determined by the shape of r which comes from model.randomness
        '''
        state = state.copy()
        sites, unifs = r
        for v, u in sites, unifs:
            S = state[self.lattice.nbr[v]].sum()
            p = self.table[(S + 4) // 2]
            state[v] = 1 if u <= p else -1
        return state
    