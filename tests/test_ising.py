'''
Test Ising model
'''

import random
import numpy as np
import pytest
from src.lattice import torus
from src.ising import Ising

@pytest.fixture
def ising_model():
    '''Ising model instance for unit tests.'''
    L = random.randint(2,10)
    M = random.randint(0, 10)

    lattice = torus(L, M) if M != 0 else torus(L)

    return Ising(lattice=lattice, beta=0.5, h=0.0)

def test_partial_order_perserved(ising_model):
    '''
    Test that if x<=y then apply(x, r) <= apply(y,r)
    '''
    n=ising_model.n

    for _ in range(10):
        x = np.random.choice([-1, 1], size=n).astype(np.int8)

        y = x.copy()
        mask = np.random.random(n) > 0.5
        y[mask] = 1

        assert ising_model.leq(x, y), 'Initial state generation failed partial order'

        depth = random.randint(1, 100)
        key = random.randint(1, 1000)
        r = ising_model.randomness(depth, key, k=5)

        x_next = ising_model.apply(x, r)
        y_next = ising_model.apply(y, r)

        assert ising_model.leq(x_next, y_next), 'Ising apply() failed partial order'

def test_randomness_reproducability(ising_model):
    '''
    Test randomness outputs the same arrays with unchanged key and depth.
    '''
    k=100
    for _ in range(5):
        depth = random.randint(1, 100)
        key = random.randint(1, 1000)

        sites1, u1 = ising_model.randomness(depth, key, k)
        sites2, u2 = ising_model.randomness(depth, key, k)

        np.testing.assert_array_equal(sites1, sites2, err_msg="Sites generation is not reproducible")
        np.testing.assert_array_equal(u1, u2, err_msg="Uniform generation is not reproducible")

def test_randomness_divergence(ising_model):
    k = 100
    for _ in range(5):
        depth = random.randint(1, 100)
        key = random.randint(1, 1000)

        sites1, u1 = ising_model.randomness(depth, key, k)
        sites2, u2 = ising_model.randomness(depth + 1, key, k) # Change depth

        sites3, u3 = ising_model.randomness(depth, key, k)
        sites4, u4 = ising_model.randomness(depth, key + 57, k) # Change key

        assert not np.array_equal(sites1, sites2)
        assert not np.array_equal(u1, u2)

        assert not np.array_equal(sites3, sites4)
        assert not np.array_equal(u3, u4)
