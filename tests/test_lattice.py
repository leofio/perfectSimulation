'''
Test Lattice dataclass and torus construction
'''

import numpy as np
import pytest
from src.lattice import Lattice, torus

def test_size_and_shape():
    lat = torus(4)
    assert lat.n_sites == 16
    assert lat.nbr.shape == (16, 4)

def test_no_self_loops():
    lat = torus(4)
    for v in range(lat.n_sites):
        assert v not in lat.nbr[v]

def test_nbrs_distinct():
    lat = torus(4)
    for v in range(lat.n_sites):
        assert len(set(lat.nbr[v])) == len(lat.nbr[v])

def test_edges():
    lat = torus(4)
    for v in range(lat.n_sites):
        for u in lat.nbr[v]:
            assert v in lat.nbr[u]

def test_nbrs_three():
    lat = torus(3)
    assert sorted(lat.nbr[0]) == [1, 2, 3, 6]

def test_rectangular():
    lat = torus(3, 5)
    assert lat.n_sites == 15
    assert lat.nbr.shape == (15, 4)
