import os
os.environ.setdefault('MPLBACKEND', 'Agg')

from perfectSim.viz import animate_pair, save
from perfectSim.models.ising import Ising
from perfectSim.random.cftp import monotone_cftp
from perfectSim.lattice import hexagonal, boundary_values
import numpy as np

out = 'viz_out'
os.makedirs(out, exist_ok=True)

lat = hexagonal(10, 8, ghosts=True)
x_max_real = (8 - 1) * np.sqrt(3) / 2.0
bd = boundary_values(
    lat, 
    lambda x, y: np.where((x < -1e-5) | (x > x_max_real + 1e-5), 1, -1)
)
model = Ising(lat, 0.4, boundary=bd)

samples, trace = monotone_cftp(model, seed=7, trace=True, k=1)

st = trace[0]['top']
sb = trace[0]['bottom']

anim = animate_pair(lat, st, lat, sb,
                    model_a=model, model_b=model,
                    boundary_a=bd, boundary_b=bd,
                    edge_rule_a='aligned', edge_rule_b='aligned',
                    title_a='top,  t={t}', title_b='bottom,  t={t}',
                    suptitle='coupled Ising chains')
save(anim, f'{out}/cftp.gif', fps=10)
