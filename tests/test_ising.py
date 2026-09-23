'''
Test Ising model
'''

import random
import numpy as np
import pytest
from src.lattice import torus
from src.models.ising import Ising

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
    B = 1

    for _ in range(10):
        x = np.random.choice([-1, 1], size=(B, n)).astype(np.int8)

        y = x.copy()
        mask = np.random.random((B, n)) > 0.5
        y[mask] = 1

        assert ising_model.leq(x, y), 'Initial state generation failed partial order'

        depth = random.randint(1, 100)

        base_sequence = np.random.SeedSequence()
        keys = base_sequence.spawn(B)
        r = ising_model.randomness(depth, keys, k=5)

        x_next = ising_model.apply(x, r)
        y_next = ising_model.apply(y, r)

        assert ising_model.leq(x_next, y_next), 'Ising apply() failed partial order'

def test_randomness_reproducability(ising_model):
    '''
    Test randomness outputs the same arrays with unchanged key and depth.
    '''
    k=100
    B=3
    for _ in range(5):
        depth = random.randint(1, 100)

        base_seq = np.random.SeedSequence(random.randint(1, 1000))
        keys = base_seq.spawn(B)

        sites1, u1 = ising_model.randomness(depth, keys, k)
        sites2, u2 = ising_model.randomness(depth, keys, k)

        np.testing.assert_array_equal(sites1, sites2, err_msg="Sites generation is not reproducible")
        np.testing.assert_array_equal(u1, u2, err_msg="Uniform generation is not reproducible")

def test_randomness_divergence(ising_model):
    k = 100
    B = 3
    for _ in range(5):
        depth = random.randint(1, 100)
        seq1 = np.random.SeedSequence(random.randint(1, 1000))
        keys1 = seq1.spawn(B)
        
        seq2 = np.random.SeedSequence(random.randint(1001, 2000))
        keys2 = seq2.spawn(B)

        sites1, u1 = ising_model.randomness(depth, keys1, k)
        sites2, u2 = ising_model.randomness(depth + 1, keys1, k) # Change depth

        sites3, u3 = ising_model.randomness(depth, keys1, k)
        sites4, u4 = ising_model.randomness(depth, keys2, k) # Change key

        assert not np.array_equal(sites1, sites2)
        assert not np.array_equal(u1, u2)

        assert not np.array_equal(sites3, sites4)
        assert not np.array_equal(u3, u4)
