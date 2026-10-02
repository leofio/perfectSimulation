import pytest
import numpy as np
from scipy.stats import chisquare

from perfectSim.lattice.lattice import make_lattice, triangular
from perfectSim.models.hardCore import HardCoreGeneral
from perfectSim.random.bounding_cftp import bounding_cftp


@pytest.fixture
def k3_lattice():
    """Triangle graph K_3: 3 sites, each connected to the other 2. Non-bipartite."""
    nbr = np.array([
        [1, 2],
        [0, 2],
        [0, 1]
    ], dtype=np.int32)
    return make_lattice(n_sites=3, nbr=nbr, is_bipartite=False)

def test_hardcore_deterministic_transitions(k3_lattice):
    activity = 1.0  # p = 0.5
    model = HardCoreGeneral(k3_lattice, activity=activity)

    # 4 distinct batches to test all branches:
    # 0: Node 0 has occupied neighbor (node 1 = 1) -> Definite Block -> 0
    # 1: Node 0 has all empty neighbors -> Definite Place -> 1
    # 2: Node 0 has all empty neighbors, but u >= p -> Definite Removal -> 0
    # 3: Node 0 has an unknown neighbor (node 1 = 2) -> Ambiguity retention -> 2
    states = np.array([
        [0, 1, 0],
        [0, 0, 0],
        [0, 0, 0],
        [0, 2, 0]
    ], dtype=np.int8)

    sites = np.array([[0], [0], [0], [0]], dtype=np.int32)
    unifs = np.array([[0.1], [0.1], [0.9], [0.1]], dtype=np.float64)

    new_states = model.apply(states, (sites, unifs))

    assert new_states[0, 0] == 0, "Site with occupied neighbor must become 0"
    assert new_states[1, 0] == 1, "Site with free neighbors and u < p must become 1"
    assert new_states[2, 0] == 0, "Site with u >= p must become 0"
    assert new_states[3, 0] == 2, "Site with unknown neighbor and u < p must remain unknown (2)"

def test_hardcore_independent_set_invariant():
    """Verify that every sample on a non-bipartite triangular grid is a valid independent set."""
    lat = triangular(3, 3, ghosts=False)
    model = HardCoreGeneral(lat, activity=0.8)

    B = 50
    samples = bounding_cftp(model, B=B, seed=42)

    # No unknown states remaining
    assert not (samples == model.unknown).any()

    # For every site with a particle, no real neighbor can have a particle
    for v in range(lat.n_sites):
        nbrs = lat.nbr[v]
        real_nbrs = nbrs[nbrs < lat.n_sites]
        
        occupied_v = samples[:, v] == 1
        adj_occupied = samples[:, real_nbrs] == 1
        
        # If site v is 1, all its neighbors must be 0
        assert not (occupied_v[:, None] & adj_occupied).any(), f"Adjacent particles found around site {v}"