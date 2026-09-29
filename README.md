# perfectSim

Exact (perfect) sampling from probabilistic models on lattices including, Ising, hard-core gas,
and the random-cluster/Potts model. Coupling-from-the-past (CFTP) based,
with a matplotlib-based visualiser for the lattices and the sampling
trajectories themselves.

Unlike standard MCMC, which only samples *approximately* after some
(unknown) burn-in, CFTP produces samples that are exactly distributed
according to the model's true stationary distribution.

<img src="viz_out/cftp.gif?v=2" alt="CFTP Animation" width="600" />

## Features

- **Lattice geometry**: square grids, tori (periodic boundaries), and
  triangular lattices, in 2D, with optional boundary/"ghost" sites for
  fixed boundary conditions.
- **Models**:
  - Ising (ferromagnetic and antiferromagnetic)
  - Hard-core gas (bipartite and general graphs)
  - Random-cluster model, with a Potts-colouring pipeline built on top
- **Exact sampling algorithms**:
  - Monotone CFTP, for models with a natural monotone coupling
  - Bounding-chain CFTP, for models (e.g. antiferromagnetic Ising, hard-core
    on non-bipartite graphs) that aren't naturally monotone
  - Read-once CFTP (ROCFTP), for drawing many samples efficiently from a
    single forward stream of randomness
- **Visualisation**: render a lattice state as vertices and edges, with
  boundary sites shown in a shaded band, and animate a sequence of states.

## Installation

```bash
git clone https://github.com/<you>/perfectSimulation.git
cd perfectSimulation
pip install -r requirements.txt   # numpy, numba, matplotlib
```

Not yet published to PyPI.

## Quickstart

```python
from perfectSim.lattice import grid, boundary_values
from perfectSim.models.ising import Ising
from perfectSim.random.cftp import monotone_cftp
from perfectSim.viz import draw

# 8x8 grid with a fixed +1 boundary
lat = grid(8, ghosts=True)
bd = boundary_values(lat, 1)

model = Ising(lat, beta=0.4, boundary=bd)
samples = monotone_cftp(model, B=1, seed=0)

draw(lat, samples[0], boundary=bd, model=model, edge_rule='aligned')
```

## Package layout

```
perfectSim/
├── lattice.py       # lattice construction: grid, torus, triangular, boundaries
├── models/
│   ├── baseModel.py     # MonotoneModel / BoundingModel base classes
│   ├── ising.py         # Ising, IsingAntiFerromagnetic
│   ├── hardCore.py       # HardCoreBipartite, HardCoreGeneral
│   ├── randomCluster.py  # MonotoneRandomCluster
│   └── potts.py          # RC -> Potts colouring
├── random/
│   ├── cftp.py           # monotone_cftp
│   ├── bounding_cftp.py  # bounding_cftp
│   └── rocftp.py         # rocftp_fixed
└── viz/
    ├── api.py            # draw, animate, save
    ├── artist.py          # matplotlib rendering
    ├── frames.py          # state -> renderable frame
    ├── palettes.py        # value -> colour
    ├── geometry.py         # hulls, edge layout
    └── trajectory.py       # throwaway forward-simulation trajectories for testing
```

## Which algorithm for which model?

| Model | Coupling | Algorithm |
|---|---|---|
| Ising, β ≥ 0 | monotone | `monotone_cftp` |
| Ising, β < 0 (antiferromagnetic) | non-monotone | `bounding_cftp` |
| Hard-core gas, bipartite graph | monotone (checkerboard trick) | `monotone_cftp` |
| Hard-core gas, general graph | non-monotone | `bounding_cftp` |
| Random-cluster / Potts | monotone | `monotone_cftp`, then `potts_from_rc` |

`rocftp_fixed` works with any `MonotoneModel` and is the more efficient
choice when drawing many samples from the same model.

## Status

This is a research/hobby project under active development, not a stable
release. In particular:

- The random-cluster/Potts pipeline requires `numba`.
- Visualisation currently supports 2D lattices only.
- No PyPI release yet; install from source.

## License

MIT — see [LICENSE](LICENSE).
