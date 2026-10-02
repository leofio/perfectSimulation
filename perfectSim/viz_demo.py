'''Renders a gallery of example images/animations into ./viz_out.'''
import os
os.environ.setdefault('MPLBACKEND', 'Agg')
import numpy as np
import networkx as nx
from perfectSim.lattice.lattice import grid, triangular, torus, boundary_values, hexagonal, complete
from perfectSim.lattice.latticenx import from_nx
from perfectSim.models.ising import Ising
from perfectSim.models.hardCore import HardCoreBipartite
from perfectSim.models.randomCluster import MonotoneRandomCluster
from perfectSim.models.potts import potts_from_rc
from perfectSim.viz import draw, animate, save, forward_trajectory, categorical_palette

out = 'viz_out'
os.makedirs(out, exist_ok=True)
rng = np.random.default_rng(0)

# 1. Ising on a grid, boundary +1 on the top/left, -1 elsewhere; solid edges = aligned spins.
lat = grid(12, ghosts=True)
bd = boundary_values(lat, lambda i, j: np.where((i < 0) | (j < 0), 1, -1))
model = Ising(lat, beta=0.44, boundary=bd)
states, upd = forward_trajectory(model, steps=1500, stride=15, seed=1)
art = draw(lat, states[-1], boundary=bd, model=model, edge_rule='aligned', title='Ising, beta=0.44')
art.fig.savefig(f'{out}/ising_grid.png', dpi=80)
save(animate(lat, states, boundary=bd, model=model, edge_rule='aligned'),
     f'{out}/ising.gif', fps=10, dpi=60)

# 2. Triangular lattice with a boundary (band follows the parallelogram).
tri = triangular(5, 3, ghosts=True)
tbd = boundary_values(tri, 1)
draw(tri, rng.choice([-1, 1], tri.n_sites), boundary=tbd, edge_rule='aligned',
     title='triangular').fig.savefig(f'{out}/triangular.png', dpi=80)

# 3. Torus: wrap-around edges drawn as stubs.
tor = torus(8)
draw(tor, rng.choice([-1, 1], tor.n_sites), edge_rule='aligned',
     title='torus').fig.savefig(f'{out}/torus.png', dpi=80)

# 4. Random-cluster edges over Potts colours (Edwards-Sokal style picture).
lat = triangular(6, ghosts=True)
q_states = 4
bd = boundary_values(lat, 0)
rc_model = MonotoneRandomCluster(lat, p=0.6, q=q_states, boundary_partitions=[list(bd.keys())])
rc_states_traj, _ = forward_trajectory(rc_model, x0=rc_model.bottom, steps=100, stride=100, seed=42)
final_rc_state = np.expand_dims(rc_states_traj[-1], axis=0)
potts_states = potts_from_rc(final_rc_state, rc_model, q=q_states, boundary=bd, seed=42)
art = draw(lat, potts_states[0], edge_states=final_rc_state[0], boundary=bd,
           palette=categorical_palette(q_states), title='RC to Potts (100 steps)')
art.fig.savefig(f'{out}/rc_potts_sim.png', dpi=80)

# 5. Hard-core gas, forward trajectory.
hc_lat = grid(10)
hc = HardCoreBipartite(hc_lat, activity=1.5)
hs, hu = forward_trajectory(hc, steps=600, stride=6, seed=2)
draw(hc_lat, hs[-1], model=hc, title='hard-core').fig.savefig(f'{out}/hardcore.png', dpi=80)
print('wrote', sorted(os.listdir(out)))

# Hexagonal Grid: Hard-Core Evolution
hex_lat = hexagonal(8, ghosts=True)
bd = boundary_values(hex_lat, lambda x, y: np.where((x + y).astype(int) % 2 == 0, 1, 0))
hc_model = HardCoreBipartite(hex_lat, activity=1.5)
states, updated = forward_trajectory(hc_model, steps=100, stride=1, seed=42)
anim = animate(hex_lat, states, boundary=bd, model=hc_model,highlight=updated, title='Hexagonal Hard-Core')
save(anim, f'{out}/hex_hardcore.gif', fps=10)


# Kn

kn = complete(12)
kn_rc = MonotoneRandomCluster(kn, p=0.4, q=1.5)
states, updated = forward_trajectory(kn_rc)
kn_anim = animate(kn, edge_states=states)
save(kn_anim, f'{out}/kn_rc.gif', fps=10)

# L-Shape
internal_coords = set(
    [(r, c) for r in range(2) for c in range(5)] + 
    [(r, c) for r in range(2, 5) for c in range(2)]
)

G = nx.Graph()
for r, c in internal_coords:
    G.add_node((r, c), pos=(r, c)) # boundary is False by default
for r, c in internal_coords:
    for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
        nr, nc = r + dr, c + dc
        
        if (nr, nc) not in internal_coords:
            if not G.has_node((nr, nc)):
                G.add_node((nr, nc), pos=(nr, nc), boundary=True)
                
        G.add_edge((r, c), (nr, nc))

l_lat = from_nx(
    G, 
    boundary_key='boundary', 
    layout=lambda g: nx.get_node_attributes(g, 'pos')
)

bd = boundary_values(
    l_lat, 
    lambda r, c: np.where((r > 2) | (c < 2), -1, 1)
)
ising_model = Ising(l_lat, beta=0.44, boundary=bd)
states, updated = forward_trajectory(ising_model, steps=400, stride=4, seed=10)

anim = animate(
    l_lat, 
    states, 
    boundary=bd, 
    model=ising_model, 
    highlight=updated, 
    edge_rule='aligned',
    wrap='straight',  
    title='Ising Model on 2x5 / 2x3 L-Shape'
)
save(anim, f'{out}/l_shape_grid_ising.gif', fps=12)
