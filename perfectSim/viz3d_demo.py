'''Renders a gallery of 3D example images/animations into ./viz3d_out.'''
import os
os.environ.setdefault('MPLBACKEND', 'Agg')
import numpy as np
from perfectSim.lattice.lattice import boundary_values
from perfectSim.lattice.lattice3d import grid3d, torus3d, complete_sphere
from perfectSim.models.ising import Ising
from perfectSim.models.hardCore import HardCoreBipartite
from perfectSim.models.randomCluster import MonotoneRandomCluster
from perfectSim.models.potts import potts_from_rc
from perfectSim.viz3d import (draw, animate, animate_pair, save, forward_trajectory,
                              categorical_palette)

out = 'viz3d_out'
os.makedirs(out, exist_ok=True)
rng = np.random.default_rng(0)

# 1. 3D Ising on a cube; +1 on the top face (i < 0), -1 on the other faces; solid edges = aligned spins.
lat = grid3d(6, ghosts=True)
bd = boundary_values(lat, lambda i, j, k: np.where(i < 0, 1, -1))
model = Ising(lat, beta=0.25, boundary=bd)          # 3D critical beta is about 0.2217
states, upd = forward_trajectory(model, steps=3000, stride=30, seed=1)
art = draw(lat, states[-1], boundary=bd, model=model, edge_rule='aligned', title='3D Ising, beta=0.25')
art.fig.savefig(f'{out}/ising3d.png', dpi=80)
save(animate(lat, states, boundary=bd, model=model, edge_rule='aligned', highlight=upd, spin=2.0), f'{out}/ising3d.gif', fps=10, dpi=60)

# 2. Same state, cut open with a slab (layers i = 0..2) so the interior is visible.
draw(lat, states[-1], boundary=bd, model=model, edge_rule='aligned', slab=(0, 0, 2),
     title='slab i in [0, 2]').fig.savefig(f'{out}/ising3d_slab.png', dpi=80)

# 3. Torus: wrap-around edges drawn as stubs.
tor = torus3d(4)
draw(tor, rng.choice([-1, 1], tor.n_sites), edge_rule='aligned', title='3D torus').fig.savefig(
    f'{out}/torus3d.png', dpi=80)

# 4. Random-cluster edges over Potts colours (Edwards-Sokal picture) in 3D.
lat = grid3d(4, ghosts=True)
q = 3
bd = boundary_values(lat, 0)
rc = MonotoneRandomCluster(lat, p=0.4, q=q, boundary_partitions=[list(bd.keys())])
rc_traj, _ = forward_trajectory(rc, x0=rc.bottom, steps=200, stride=200, seed=42)
rc_state = np.expand_dims(rc_traj[-1], axis=0)
potts = potts_from_rc(rc_state, rc, q=q, boundary=bd, seed=42)
draw(lat, potts[0], edge_states=rc_state[0], boundary=bd, palette=categorical_palette(q),
     title='RC to Potts (3D)').fig.savefig(f'{out}/rc_potts3d.png', dpi=80)

# 5. Hard-core gas on the cubic lattice.
hl = grid3d(6)
hc = HardCoreBipartite(hl, activity=1.5)
hs, hu = forward_trajectory(hc, steps=1000, stride=10, seed=2)
draw(hl, hs[-1], model=hc, title='3D hard-core').fig.savefig(f'{out}/hardcore3d.png', dpi=80)

# 6. K_n on a sphere.
kn = complete_sphere(12)
kn_rc = MonotoneRandomCluster(kn, p=0.4, q=1.5)
kn_states, _ = forward_trajectory(kn_rc)
save(animate(kn, edge_states=kn_states, wrap='straight', spin=3.0), f'{out}/kn_rc3d.gif', fps=10)

# 7. Two chains side by side, lock-step (top-started vs bottom-started).
lat = grid3d(5, ghosts=True)
bd = boundary_values(lat, 1)
m = Ising(lat, beta=0.25, boundary=bd)
sa, _ = forward_trajectory(m, x0=m.top, steps=600, stride=6, seed=3)
sb, _ = forward_trajectory(m, x0=m.bottom, steps=600, stride=6, seed=3)
save(animate_pair(lat, sa, lat, sb, title_a='top start t={t}', title_b='bottom start t={t}',
                  boundary_a=bd, boundary_b=bd, model_a=m, model_b=m,
                  edge_rule_a='aligned', edge_rule_b='aligned'),
     f'{out}/pair3d.gif', fps=10, dpi=50)

print('wrote', sorted(os.listdir(out)))
