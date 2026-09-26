'''
Test Random Cluster model.
'''

import random
import numpy as np
import pytest
from src.lattice import torus, grid
from src.models.randomCluster import MonotoneRandomCluster

@pytest.fixture
def rc_model():
    L, M = random.randint(3, 6), random.randint(3, 6)
    lattice = grid(L, M)
    return MonotoneRandomCluster(lattice=lattice, p=0.5, q=2.0)

def test_partial_order_preserved(rc_model):
    n = rc_model.n
    for _ in range(10):
        x = np.random.choice([0, 1], size=(1, n)).astype(np.int8)
        y = x.copy()
        mask = np.random.random((1, n)) > 0.5
        y[mask] = 1
        assert rc_model.leq(x, y)
        r = rc_model.randomness(random.randint(1, 100), np.random.SeedSequence().spawn(1), k=5)
        assert rc_model.leq(rc_model.apply(x, r), rc_model.apply(y, r))


def test_randomness_reproducability(rc_model):
    '''
    Test randomness outputs the same arrays with unchanged key and depth.
    '''
    k=100
    B=3
    for _ in range(5):
        depth = random.randint(1, 100)

        base_seq = np.random.SeedSequence(random.randint(1, 1000))
        keys = base_seq.spawn(B)

        sites1, u1 = rc_model.randomness(depth, keys, k)
        sites2, u2 = rc_model.randomness(depth, keys, k)

        np.testing.assert_array_equal(sites1, sites2, err_msg="Sites generation is not reproducible")
        np.testing.assert_array_equal(u1, u2, err_msg="Uniform generation is not reproducible")

def test_randomness_divergence(rc_model):
    k = 100
    B = 3
    for _ in range(5):
        depth = random.randint(1, 100)
        seq1 = np.random.SeedSequence(random.randint(1, 1000))
        keys1 = seq1.spawn(B)
        
        seq2 = np.random.SeedSequence(random.randint(1001, 2000))
        keys2 = seq2.spawn(B)

        sites1, u1 = rc_model.randomness(depth, keys1, k)
        sites2, u2 = rc_model.randomness(depth + 1, keys1, k) # Change depth

        sites3, u3 = rc_model.randomness(depth, keys1, k)
        sites4, u4 = rc_model.randomness(depth, keys2, k) # Change key

        assert not np.array_equal(sites1, sites2)
        assert not np.array_equal(u1, u2)

        assert not np.array_equal(sites3, sites4)
        assert not np.array_equal(u3, u4)

def test_connected_excluding_direct():
    lat = grid(2, 2)  # 4-cycle: edges 0-1, 0-2, 1-3, 2-3
    model = MonotoneRandomCluster(lat, p=0.5, q=2.0)

    def eid(a, b):
        return next(i for i, (x, y) in enumerate(lat.edges)
                     if set((int(x), int(y))) == {a, b})

    e01 = eid(0, 1)
    e02 = eid(0, 2)
    e13 = eid(1, 3)
    e23 = eid(2, 3)

    cfg = np.zeros(lat.n_edges, dtype=np.int8)
    cfg[e02] = 1
    cfg[e23] = 1
    cfg[e13] = 1  # long way round: 0-2-3-1

    assert model._connected_excluding(cfg, 0, 1, e01)

    cfg[e23] = 0
    assert not model._connected_excluding(cfg, 0, 1, e01)

def test_q_equals_one_is_plain_percolation():
    lat = grid(4, 4)
    model = MonotoneRandomCluster(lat, p=0.3, q=1.0)
    assert model.p_merge == pytest.approx(model.p)#

def test_boundary_partition_forces_connection():
    '''
    Wiring all ghost ids into one boundary_partitions group should let two
    real sites be seen as connected purely via the boundary, even with no
    open edges between them directly.
    '''
    lat = grid(2, 2, ghosts=True)
    ghost_ids = list(range(lat.n_sites, lat.null))
    assert len(ghost_ids) >= 2, 'test needs at least two ghost sites to be meaningful'

    model = MonotoneRandomCluster(
        lat, p=0.5, q=2.0,
        boundary_partitions=[ghost_ids],
    )

    boundary_edges_by_real_site = {}
    for eid, (a, b) in enumerate(lat.edges):
        a, b = int(a), int(b)
        if a >= lat.n_sites and b < lat.n_sites:
            real_site, ghost = b, a
        elif b >= lat.n_sites and a < lat.n_sites:
            real_site, ghost = a, b
        else:
            continue
        boundary_edges_by_real_site.setdefault(real_site, []).append(eid)

    real_sites_with_boundary_edges = list(boundary_edges_by_real_site.keys())
    assert len(real_sites_with_boundary_edges) >= 2, \
        'test lattice needs at least two distinct real sites touching the boundary'

    site_x, site_y = real_sites_with_boundary_edges[0], real_sites_with_boundary_edges[1]
    e_x = boundary_edges_by_real_site[site_x][0]
    e_y = boundary_edges_by_real_site[site_y][0]

    cfg = np.zeros(lat.n_edges, dtype=np.int8)
    cfg[e_x] = 1
    cfg[e_y] = 1

    # With ghosts wired together, site_x -- ghost -- (wired) -- ghost -- site_y
    # should read as connected once both boundary edges are open.
    # Use a probe edge id that isn't e_x or e_y (any other edge index works,
    # since _connected_excluding only excludes the passed edge from the search).
    probe_e = next(eid for eid in range(lat.n_edges) if eid not in (e_x, e_y))
    assert model._connected_excluding(cfg, site_x, site_y, probe_e)

    # Sanity check: without the boundary_partitions wiring, the same cfg
    # should NOT connect site_x and site_y (distinct, un-identified ghosts).
    unwired_model = MonotoneRandomCluster(lat, p=0.5, q=2.0)  # no boundary_partitions
    assert not unwired_model._connected_excluding(cfg, site_x, site_y, probe_e)


def test_apply_from_top_and_bottom_no_crash():
    '''
    Smoke test: apply() should run without indexing errors starting from both
    top (all edges open) and bottom (all edges closed), including lattices
    with a boundary, so that every boundary edge gets exercised at least once.
    '''
    lat = grid(3, 3, ghosts=True)
    model = MonotoneRandomCluster(lat, p=0.5, q=2.0)

    B = 2
    k = 50  # large enough that boundary edges are very likely to be sampled

    base_sequence = np.random.SeedSequence(0)
    keys = base_sequence.spawn(B)
    r = model.randomness(depth=1, keys=keys, k=k)

    top_states = np.tile(model.top, (B, 1))
    bottom_states = np.tile(model.bottom, (B, 1))

    top_result = model.apply(top_states, r)
    bottom_result = model.apply(bottom_states, r)

    for result in (top_result, bottom_result):
        assert result.shape == (B, model.n)
        assert result.dtype == np.int8
        assert set(np.unique(result)).issubset({0, 1})
