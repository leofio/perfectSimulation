'''
Pyvista demo
'''

import numpy as np
import pyvista as pv
import networkx as nx
from pathlib import Path

from perfectSim.viz3dpyvista.api import draw, animate
from perfectSim.models.ising import Ising
from perfectSim.lattice.latticenx import from_nx

out_dir = Path("pyvista_out")
out_dir.mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(seed=42)

# 3D cubic

lat_nx = nx.grid_graph(dim=[5,5,7])
lat_cubic = from_nx(lat_nx)
values = rng.choice([-1, 1], lat_cubic.n_sites)
plotter = draw(lat_cubic, values)

cubic_img_path = out_dir / "3d_cubic.png"
plotter.show(screenshot=str(cubic_img_path))
print(f"Saved cubic plot to {cubic_img_path}")

# Hex 3D

hex_2d = nx.hexagonal_lattice_graph(3, 3)
z_axis = nx.path_graph(4)
stacked_hex_3d = nx.cartesian_product(hex_2d, z_axis)

lat_hex = from_nx(stacked_hex_3d)
ising_model = Ising(lat_hex, 0.4)
states = np.full((200, lat_hex.n_sites), 1, dtype=np.int32)
for i in range(1, 200):
    r = ising_model.randomness(i, [42], 1)
    states[i] = ising_model.apply(states[i-1], r)

hex_gif_path = out_dir / "3d_ising.gif"
animate(lat_hex, states, str(hex_gif_path))
print(f"Saved animation to {hex_gif_path}")
