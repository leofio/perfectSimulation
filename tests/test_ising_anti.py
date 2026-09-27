import pytest
import numpy as np
from scipy.stats import chisquare

from src.lattice import make_lattice, torus
from src.models.ising import IsingAntiFerromagnetic
from src.perfectSim.bounding_cftp import bounding_cftp


@pytest.fixture
def k3_lattice():
    """Triangle graph K_3: 3 sites, each connected to the other 2. Non-bipartite."""
    nbr = np.array([
        [1, 2],
        [0, 2],
        [0, 1]
    ], dtype=np.int32)
    return make_lattice(n_sites=3, nbr=nbr, is_bipartite=False)

def test_ising_deterministic_transitions(k3_lattice):
    beta = -0.5
    model = IsingAntiFerromagnetic(k3_lattice, beta=beta, h=0.0)

    # State 0: Node 0 neighbors are (+1, +1) -> S_min = 2, S_max = 2
    # State 1: Node 0 neighbors are (-1, -1) -> S_min = -2, S_max = -2
    # State 2: Node 0 neighbors are (0, 0) (unknown) -> S_min = -2, S_max = 2
    states = np.array([
        [model.unknown, 1, 1],
        [model.unknown, -1, -1],
        [model.unknown, model.unknown, model.unknown]
    ], dtype=np.int8)

    sites = np.array([[0], [0], [0]], dtype=np.int32)

    # For S=2: p = 1 / (1 + exp(2)) ≈ 0.1192
    # For S=-2: p = 1 / (1 + exp(-2)) ≈ 0.8808
    p_min = model.table[2 + k3_lattice.max_degree]
    p_max = model.table[-2 + k3_lattice.max_degree]

    unifs = np.array([
        [0.05],  # u < p_min -> +1
        [0.95],  # u > p_max -> -1
        [0.50]   # p_min < u < p_max -> unknown (0)
    ], dtype=np.float64)

    new_states = model.apply(states, (sites, unifs))

    assert new_states[0, 0] == 1, "Must collapse to +1 when u <= p_min"
    assert new_states[1, 0] == -1, "Must collapse to -1 when u > p_max"
    assert new_states[2, 0] == model.unknown, "Must remain unknown (0) when p_min < u <= p_max"

def test_ising_torus_coalescence_and_invariants():
    """Verify that a 3x3 non-bipartite torus completely coalesces to +-1 spins."""
    # 3x3 torus is not bipartite because 3 is odd
    lat = torus(3, 3)
    assert not lat.is_bipartite

    model = IsingAntiFerromagnetic(lat, beta=-0.3, h=0.0)

    B = 30
    samples = bounding_cftp(model, B=B, seed=42)

    # Assert all unknowns (0) have been eliminated
    assert not (samples == model.unknown).any()
    # Assert every site is strictly in {-1, 1}
    assert set(np.unique(samples)).issubset({-1, 1})