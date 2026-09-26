'''
Random Cluster model.
'''

import numpy as np
from src.lattice import Lattice
from src.models.baseModel import MonotoneModel

class MonotoneRandomCluster(MonotoneModel):
    def __init__(self, lattice: Lattice, p: float, q: float, boundary_partitions=None):
        super().__init__(lattice)
        assert q >= 1, 'Monotone RC requires q >= 1'
        assert (1 >= p) and (p >= 0), 'Require 1 >= p >= 0'
        assert lattice.edges is not None, 'RC model requires a lattice with an edge set'

        self.p = p
        self.p_merge = p / (p + q*(1 - p))

        self.boundary_partitions = boundary_partitions or [] # each entry is an iterable of ghost ids to be identified to one cluster
        self._base_parent = self._build_base_parent()

    @property
    def n(self): return self.lattice.n_edges

    @property
    def top(self): return np.ones(self.n, dtype=np.int8)

    @property
    def bottom(self): return np.zeros(self.n, dtype=np.int8)

    def leq(self, x, y): return bool(np.all(x <= y))

    def randomness(self, depth, keys, k):
            '''
            Get the randomness for k steps over a batch of keys.
            keys: (batch_size,) array of seed keys
            Returns sites and unifs of shape (batch_size, k)
            '''
            batch_size = len(keys)
            sites = np.empty((batch_size, k), dtype=np.int32)
            unifs = np.empty((batch_size, k), dtype=np.float64)
    
            for i, key in enumerate(keys):
                rng = np.random.Generator(np.random.Philox(seed=key, counter=depth))
                sites[i] = rng.integers(0, self.n, k)
                unifs[i] = rng.random(k)
    
            return sites, unifs

    def _build_base_parent(self):
        parent = np.arange(self.lattice.null)
        for group in self.boundary_partitions:
            group = list(group)
            root = group[0]
            for g in group[1:]:
                parent[g] = root
        return parent

    def _connected_excluding(self, cfg, u, v, e):
        parent = self._base_parent.copy()

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        open_edges = np.flatnonzero(cfg)
        for eid in open_edges:
            if eid == e: continue
            a, b_ = self.lattice.edges[eid]
            ra, rb = find(a), find(b_)
            if ra != rb: parent[ra] = rb
        return bool(find(u) == find (v))

    def apply(self, states, r) -> np.ndarray:
        edges_, unifs = r
        batch_size, k = edges_.shape
        states = states.copy()

        for b in range(batch_size):
            cfg = states[b]
            for step in range(k):
                e = edges_[b, step]
                u, v = self.lattice.edges[e]
                connected = self._connected_excluding(cfg, u, v, e)
                p_open = self.p if connected else self.p_merge
                cfg[e] = 1 if unifs[b, step] < p_open else 0
        return states
