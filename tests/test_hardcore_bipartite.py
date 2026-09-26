'''
Test Hard Core Bipartite
'''
import random
import numpy as np
import pytest
from src.lattice import grid, boundary_values
from src.models.hardCore import HardCoreBipartite

@pytest.fixture
def hc_model():
    L, M = random.randint(3, 10), random.randint(3, 10)
    lattice = grid(L, M)
    return HardCoreBipartite(lattice=lattice, activity=1.0)

def test_partial_order_preserved(hc_model):
    n = hc_model.n
    colour = hc_model.lattice.colour

    for _ in range(10):
        x = np.random.choice([0, 1], size=(1, n)).astype(np.int8)
        y = x.copy()
        mask = (np.random.random((1, n)) > 0.5)[0]

        move_up = mask & (colour == 0)
        move_down = mask & (colour == 1)
        y[0, move_up] = 1
        y[0, move_down] = 0

        assert hc_model.leq(x, y), 'Initial state generation failed partial order'

        depth = random.randint(1, 100)
        r = hc_model.randomness(depth, np.random.SeedSequence().spawn(1), k=5)

        x_next = hc_model.apply(x, r)
        y_next = hc_model.apply(y, r)

        assert hc_model.leq(x_next, y_next), 'HardCore apply() failed partial order'

def test_randomness_reproducability(hc_model):
    '''
    Test randomness outputs the same arrays with unchanged key and depth.
    '''
    k=100
    B=3
    for _ in range(5):
        depth = random.randint(1, 100)

        base_seq = np.random.SeedSequence(random.randint(1, 1000))
        keys = base_seq.spawn(B)

        sites1, u1 = hc_model.randomness(depth, keys, k)
        sites2, u2 = hc_model.randomness(depth, keys, k)

        np.testing.assert_array_equal(sites1, sites2, err_msg="Sites generation is not reproducible")
        np.testing.assert_array_equal(u1, u2, err_msg="Uniform generation is not reproducible")

def test_randomness_divergence(hc_model):
    k = 100
    B = 3
    for _ in range(5):
        depth = random.randint(1, 100)
        seq1 = np.random.SeedSequence(random.randint(1, 1000))
        keys1 = seq1.spawn(B)
        
        seq2 = np.random.SeedSequence(random.randint(1001, 2000))
        keys2 = seq2.spawn(B)

        sites1, u1 = hc_model.randomness(depth, keys1, k)
        sites2, u2 = hc_model.randomness(depth + 1, keys1, k) # Change depth

        sites3, u3 = hc_model.randomness(depth, keys1, k)
        sites4, u4 = hc_model.randomness(depth, keys2, k) # Change key

        assert not np.array_equal(sites1, sites2)
        assert not np.array_equal(u1, u2)

        assert not np.array_equal(sites3, sites4)
        assert not np.array_equal(u3, u4)

def test_boundary_validation():
    """Ensure the model rejects invalid boundary specifications."""
    lat = grid(3, 3, ghosts=True)
    
    bad_spins = boundary_values(lat, 2)
    with pytest.raises(AssertionError, match=r"Hard-core boundary must be 0s or 1s"):
        HardCoreBipartite(lat, activity=0.5, boundary=bad_spins)
        
    with pytest.raises(AssertionError, match="Boundary keys must be ghost ids"):
        HardCoreBipartite(lat, activity=1.0, boundary={0: 1})

def test_bvals_array_construction():
    """Verify that the self.bvals array maps ghost states and the null slot correctly."""
    lat = grid(3, 3, ghosts=True)
    
    b_dict = boundary_values(lat, 1)
    model = HardCoreBipartite(lat, activity=1.0, boundary=b_dict)
    
    assert model.bvals[-1] == 0 
    assert np.all(model.bvals[:-1] == 1)
    assert len(model.bvals) == lat.n_boundary + 1

def test_exclusion_constraint_maintained(hc_model):
    states = np.zeros((1, hc_model.n), dtype=np.int8)
    r = hc_model.randomness(1, np.random.SeedSequence().spawn(1), k=200)
    states = hc_model.apply(states, r)
    for v in range(hc_model.n):
        for u in hc_model.lattice.nbr[v]:
            if u < hc_model.n:
                assert not (states[0, v] == 1 and states[0, u] == 1)

def test_top_bottom_satisfy_exclusion(hc_model):
    for state in (hc_model.top, hc_model.bottom):
        for v in range(hc_model.n):
            for u in hc_model.lattice.nbr[v]:
                if u < hc_model.n:
                    assert not (state[v] == 1 and state[u] == 1)
