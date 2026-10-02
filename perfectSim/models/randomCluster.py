'''
Random Cluster model.
'''

import numpy as np
from numba import njit
from perfectSim.lattice.lattice import Lattice
from perfectSim.models.baseModel import MonotoneModel

@njit(cache=True)
def jit_find(parent, x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x

@njit(cache=True)
def jit_connected_excluding(cfg, edges, base_parent, u, v, e):
    parent = base_parent.copy()
    n_edges = cfg.shape[0]
    for eid in range(n_edges):
        if eid == e or cfg[eid] == 0:
            continue
        a = edges[eid, 0]
        b = edges[eid, 1]
        ra = jit_find(parent, a)
        rb = jit_find(parent, b)
        if ra != rb:
            parent[ra] = rb

    return jit_find(parent, u) == jit_find(parent, v)

@njit(cache=True)
def jit_apply(states, edge_ids, unifs, edges, base_parent, p, p_merge):
    '''
    states: (batch_size, n_edges) int8, mutated in place and returned
    edge_ids: (batch_size, k) int32, sampled edge index per step
    unifs: (batch_size, k) float64, sampled uniforms per step
    edges: (n_edges, 2) int32
    base_parent: (null,) int32
    '''
    batch_size, k = edge_ids.shape

    for b in range(batch_size):
        cfg = states[b]
        for step in range(k):
            e = edge_ids[b, step]
            u_site = edges[e, 0]
            v_site = edges[e, 1]

            connected = jit_connected_excluding(cfg, edges, base_parent, u_site, v_site, e)
            p_open = p if connected else p_merge

            cfg[e] = 1 if unifs[b, step] < p_open else 0

    return states

class MonotoneRandomCluster(MonotoneModel):
    def __init__(self, lattice: Lattice, p: float, q: float, boundary_partitions=None):
        super().__init__(lattice)
        assert q >= 1, 'Monotone RC requires q >= 1'
        assert (1 >= p) and (p >= 0), 'Require 1 >= p >= 0'
        assert lattice.edges is not None, 'RC model requires a lattice with an edge set'

        self.p = p
        self.q = q
        self.p_merge = p / (p + q*(1 - p))

        self.boundary_partitions = boundary_partitions or [] # each entry is an iterable of ghost ids to be identified to one cluster
        self.base_parent = self._build_base_parent()

        self._edges_arr = np.asarray(self.lattice.edges, dtype=np.int32)

    @property
    def n(self): return self.lattice.n_edges

    @property
    def top(self): return np.ones(self.n, dtype=np.int8)

    @property
    def bottom(self): return np.zeros(self.n, dtype=np.int8)

    def leq(self, x, y): return bool(np.all(x <= y))

    def _build_base_parent(self):
        parent = np.arange(self.lattice.null)
        for group in self.boundary_partitions:
            group = list(group)
            root = group[0]
            for g in group[1:]:
                parent[g] = root
        return parent

    def _connected_excluding(self, cfg, u, v, e):
        cfg = np.asarray(cfg, dtype=np.int8)
        return bool(
            jit_connected_excluding(
            cfg,
            self._edges_arr,
            self.base_parent,
            np.int32(u),
            np.int32(v), 
            np.int32(e)
                )
            )


    def apply(self, states, r) -> np.ndarray:
        edge_ids, unifs = r
        states = states.copy()
        edge_ids = np.ascontiguousarray(edge_ids, dtype=np.int32)
        unifs = np.ascontiguousarray(unifs, dtype=np.float64)

        return jit_apply(
            states, 
            edge_ids,
            unifs,
            self._edges_arr,
            self.base_parent,
            self.p,
            self.p_merge
            )
