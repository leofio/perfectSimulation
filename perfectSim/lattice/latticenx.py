'''
Lattice constructor form networkx object.
'''

import numpy as np
import networkx as nx
from perfectSim.lattice.lattice import Lattice, make_lattice, FREE

def from_nx(G:nx.Graph, boundary_key='boundary', layout:callable = None) -> Lattice:
    boundary_dict = nx.get_node_attributes(G, boundary_key)

    real_nodes = [n for n in G.nodes if not boundary_dict.get(n, False)]
    boundary_nodes = [n for n in G.nodes if boundary_dict.get(n, False)]
    n_sites = len(real_nodes)
    n_boundary = len(boundary_nodes)

    node_mapping = {old: new for new, old in enumerate(real_nodes + boundary_nodes)}
    G_int = nx.relabel_nodes(G, node_mapping)

    max_degree = max((d for n, d in G.degree), default=0)

    nbr = np.full((n_sites, max_degree), FREE, dtype=np.int32)

    for v in range(n_sites):
        nbrs = list(G_int.neighbors(v))
        nbr[v, :len(nbrs)] = nbrs

    is_bipartite = nx.is_bipartite(G_int)

    colour = None
    if is_bipartite and n_sites > 0:
        color_dict = nx.bipartite.color(G_int)
        colour = np.array([color_dict[n] for n in range(n_sites)], dtype=np.int8)

    coords = nx.spring_layout(G_int) if layout is None else layout(G_int)
    pos = np.array([coords[n] for n in range(n_sites + n_boundary)], dtype=np.float32)

    return make_lattice(
        n_sites=n_sites,
        nbr=nbr,
        is_bipartite=is_bipartite,
        colour=colour,
        n_boundary=n_boundary,
        pos=pos
    )
