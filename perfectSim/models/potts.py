'''
Convert a Random Cluster state to a Potts state.
'''

import numpy as np
from numba import njit
from src.models.randomCluster import jit_find, MonotoneRandomCluster

@njit(cache=True)
def jit_colour_clusters(rc_states, edges, n_sites, n_total, q, seed_array, base_parent, fixed_colours):
    '''
    rc_states: (batch_size, n_edges) int8, output from monotone_cftp
    edges: (n_edges, 2) int32
    n_sites: int
    n_total: int, lattice.null - total nodes including ghosts, sizes DSU
    q: int, number of Potts states
    seed_array: (batch_size, n_total) float64, uniforms for coloring - must include ghost sites
    base_parent: (null_size,) int32, initial wiring of boundaries
    fixed_colors: (null_size,) int32, -1 for free, or [0, q-1] for boundaries
    '''
    batch_size, n_edges = rc_states.shape
    potts_samples = np.empty((batch_size, n_sites), dtype=np.int32)

    for b in range(batch_size):
        cfg = rc_states[b]
        parent = base_parent.copy()
        for e in range(n_edges):
            if cfg[e] == 1:
                u = edges[e, 0]
                v = edges[e, 1]
                root_u = jit_find(parent, u)
                root_v = jit_find(parent, v)
                if root_u != root_v:
                    parent[root_u] = root_v

        fixed_for_root = np.full(n_total, -1, dtype=np.int32)
        for i in range(n_total):
            if fixed_colours[i] != -1:
                root = jit_find(parent, i)
                if fixed_for_root[root] == -1:
                    fixed_for_root[root] = fixed_colours[i]
                elif fixed_for_root[root] != fixed_colours[i]:
                    raise ValueError("Conflicting fixed boundary colours merged into one RC cluster")

        cluster_colours = np.empty(n_total, dtype=np.int32)
        for i in range(n_total):
            if parent[i] == i:
                if fixed_for_root[i] != -1:
                    cluster_colours[i] = fixed_for_root[i]
                else:
                    cluster_colours[i] = int(seed_array[b, i] * q)

        for i in range(n_sites):
            root = jit_find(parent, i)
            potts_samples[b, i] = cluster_colours[root]

    return potts_samples

def potts_from_rc(states, model: MonotoneRandomCluster, q, boundary, seed: int = None):
    '''
    Generate a Potts colouring from an RC configuration.
    boundary: dict, mapping ghost_id -> fixed_spin in {0, 1, ..., q-1}
    '''
    batch_size = states.shape[0]
    fixed_colours = np.full(model.lattice.null, -1, dtype=np.int32)
    for ghost, spin in boundary.items():
        fixed_colours[ghost] = spin

    base_sequence = np.random.SeedSequence(seed)
    color_rng = np.random.default_rng(base_sequence.spawn(1)[0])
    colour_unifs = color_rng.random((batch_size, model.lattice.null))

    potts = jit_colour_clusters(
        states,
        model._edges_arr,
        model.lattice.n_sites,
        model.lattice.null,
        q,
        colour_unifs,
        model.base_parent,
        fixed_colours
    )

    return potts
