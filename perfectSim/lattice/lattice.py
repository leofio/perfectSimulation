'''
Lattices
'''

from dataclasses import dataclass, replace
from typing import Optional, Union
import numpy as np

FREE = -128

@dataclass(frozen=True, eq=False)
class Lattice:
    '''
    n_sites: int, number of vertices
    nbr: np.ndarray, (n_sites, max_degree) nbr[v] contains the neighbours of v
    is_bipartite: bool, is the graph bipartite
    n_boundary: Optional[int], number of boundary vertices
    colour: Optional[np.ndarray], (n_sites,) array of 1s and 0s, required for bipartite graphs
    n_edges: Optional[int], number of edges
    edges: Optonal[np.ndarray], (n_edges, 2) mapping edge_id -> (site_u, site_v)
    pos: Optional[np.ndarray], (n_total, n_dim), coords for each vertex including ghosts
    '''
    n_sites: int
    nbr: np.ndarray
    is_bipartite: bool

    n_boundary: Optional[int] = 0
    colour: Optional[np.ndarray] = None

    n_edges: Optional[int] = 0
    edges: Optional[np.ndarray] = None

    pos: Optional[np.ndarray] = None

    def __post_init__(self):
        if self.is_bipartite:
            assert self.colour is not None, 'A 2-colouring array is required for a bipartite graph.'
            assert len(self.colour) == self.n_sites, f"Color array length ({len(self.colour)}) must match number of sites ({self.n_sites})."
            assert set(np.unique(self.colour)).issubset({0, 1}), "Color array must only contain 0s and 1s."

        if self.edges is not None:
            assert self.edges.shape == (self.n_edges, 2)

    @property
    def max_degree(self) -> int: return self.nbr.shape[1]

    @property
    def null(self) -> int: return self.n_sites + self.n_boundary

    @property
    def ghost_pos(self):
        if self.n_boundary == 0 or self.pos is None:
            return None
        return self.pos[self.n_sites : self.n_sites + self.n_sites + self.n_boundary]

def make_lattice(n_sites, nbr, is_bipartite, colour = None, n_boundary=0, pos=None) -> Lattice:
    nbr = np.asarray(nbr, dtype=np.int32)
    null = n_sites + n_boundary
    nbr = np.where(nbr == FREE, null, nbr).astype(np.int32)

    if is_bipartite:
        assert colour is not None, 'A bipartite graph requires a valid 2-colouring'

    assert nbr.shape[0] == n_sites
    assert ((nbr >= 0) & (nbr <= null)).all(), 'neighbour id out of range'
    assert not (nbr == np.arange(n_sites)[:, None]).any(), 'self-loop'

    real = [(v, u) for v in range(n_sites) for u in nbr[v] if u < n_sites]
    assert sorted(real) == sorted((u, v) for v, u in real), 'real-site adjacency is not symmetric'

    edge_set = set()
    for v in range(n_sites):
        for u in nbr[v]:
            if u != null:
                edge_set.add(tuple(sorted((v, u))))
    edges = np.array(sorted(list(edge_set)), dtype=np.int32).reshape(-1, 2)
    n_edges = len(edges)

    return Lattice(
        n_sites=n_sites, 
        nbr=nbr,
        is_bipartite=is_bipartite,
        colour=colour,
        n_boundary=n_boundary, 
        n_edges=n_edges,
        edges=edges,
        pos=pos,
    )

def _grid(L, M, offsets, is_bipartite, colour=None, periodic=False, ghosts=False):
    '''
    Create a grid lattice with option to include a boundary.
    '''
    n = L * M
    ghost = {}

    def nid(i, j):
        if periodic:
            return (i % L) * M + (j % M)
        if 0 <= i < L and 0 <= j < M:
            return i * M + j
        if not ghosts:
            return FREE
        if (i, j) not in ghost:
            ghost[(i, j)] = n + len(ghost)
        return ghost[(i, j)]

    def offs(i, j):
        return offsets(i, j) if callable(offsets) else offsets

    nbr = np.array([[nid(i + di, j + dj) for di, dj in offs(i, j)]
                    for i in range(L) for j in range(M)], dtype=np.int32)

    site_pos_list = [(i, j) for i in range(L) for j in range(M)]
    ghost_pos_list = sorted(ghost, key=ghost.get)
    full_pos = np.array(site_pos_list + ghost_pos_list, dtype=np.float32)
    return make_lattice(n, nbr, is_bipartite, colour=colour, n_boundary=len(ghost), pos=full_pos)

SQUARE = [(-1, 0), (1, 0), (0, -1), (0, 1)]
TRIANGULAR = SQUARE + [(-1, 1), (1, -1)]

def torus(L, M=None) -> Lattice:
    '''Create an LxM torus lattice'''
    M = L if M is None else M
    if (L % 2) == 0 and (M % 2) == 0:
        is_bipartite = True
        colour = np.array([(i + j) % 2 for i in range(L) for j in range(M)], dtype=np.int8)
    else:
        is_bipartite = False
        colour = None
    return _grid(L, M, SQUARE, is_bipartite, colour=colour, periodic=True)

def grid(L, M=None, ghosts=False):
    '''LxM square gird.'''
    M = L if M is None else M
    colour = np.array([(i + j) % 2 for i in range(L) for j in range(M)], dtype=np.int8)
    return _grid(L, M, SQUARE, is_bipartite=True, colour=colour, ghosts=ghosts)

def triangular(L, M=None, ghosts=False):
    '''LxM triangular grid.'''
    M = L if M is None else M
    lat = _grid(L, M, TRIANGULAR, is_bipartite=False, ghosts=ghosts)

    transform = np.array([
        [1.0, 0.0],
        [0.5, np.sqrt(3) / 2.0]
    ], dtype=np.float32)
    
    new_pos = lat.pos @ transform
        
    return replace(lat, pos=new_pos)

def boundary_values(lat: Lattice, spec: Union[int, callable]):
    '''
    spec: int, or callable(*coords) -> array of values (FREE entries skipped).
    Returns {ghost_id: value} for passing to a model.
    '''
    if lat.n_boundary == 0: return {}
    if callable(spec):
        coords = lat.ghost_pos.T
        vals = spec(*coords)
    else:
        vals = np.broadcast_to(spec, lat.n_boundary)
    return {lat.n_sites + g: int(v) for g, v in enumerate(vals) if v != FREE}

def _hex_offsets(i, j):
    vertical = (1, 0) if (i + j) % 2 == 0 else (-1, 0)
    return [(0, -1), (0, 1), vertical]

def _hex_transform(p):
    '''Map brick-wall (i, j) coords to true honeycomb coords (unit bond length).'''
    i, j = p[:, 0], p[:, 1]
    odd = np.mod(i + j, 2)
    x = j * np.sqrt(3) / 2.0
    y = 1.5 * i - 0.5 * odd
    return np.stack([x, y], axis=1).astype(np.float32)

def hexagonal(L, M=None, ghosts=False):
    '''LxM hexagonal (honeycomb) grid, stored as a brick-wall lattice.'''
    M = L if M is None else M
    colour = np.array([(i + j) % 2 for i in range(L) for j in range(M)], dtype=np.int8)
    lat = _grid(L, M, _hex_offsets, is_bipartite=True, colour=colour, ghosts=ghosts)

    new_pos = _hex_transform(lat.pos)

    return replace(lat, pos=new_pos)

def complete(n):
    '''Complete graph K_n: every site is a neighbour of every other site.'''
    idx = np.arange(n)
    nbr = np.array([np.delete(idx, v) for v in range(n)], dtype=np.int32).reshape(n, n - 1)

    # K_n is bipartite only for n <= 2
    is_bipartite = n <= 2
    colour = (idx % 2).astype(np.int8) if is_bipartite else None

    # place sites evenly on a unit circle, useful for plotting
    theta = 2 * np.pi * idx / max(n, 1)
    pos = np.stack([np.cos(theta), np.sin(theta)], axis=1).astype(np.float32)

    return make_lattice(n, nbr, is_bipartite, colour=colour, pos=pos)
