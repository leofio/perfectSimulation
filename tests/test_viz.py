import os
os.environ.setdefault('MPLBACKEND', 'Agg')

import numpy as np
import pytest
import matplotlib.pyplot as plt

from perfectSim.lattice.lattice import grid, triangular, torus, boundary_values, FREE
from perfectSim.viz import (draw, animate, save, Palette, ISING, categorical_palette,
                     default_palette, forward_trajectory, ghost_values_from_boundary,
                     rule_aligned)
from perfectSim.viz.geometry import convex_hull, padded_hull, edge_segments, orient


@pytest.fixture(autouse=True)
def _close_figs():
    yield
    plt.close('all')


# ---------- geometry ----------

def test_convex_hull_square_ignores_interior():
    pts = np.array([[0, 0], [2, 0], [2, 2], [0, 2], [1, 1], [1, 0]], dtype=float)
    assert len(convex_hull(pts)) == 4


def test_padded_hull_grows_by_radius():
    pts = np.array([[0, 0], [4, 0], [4, 4], [0, 4]], dtype=float)
    h = padded_hull(pts, 0.5)
    assert np.isclose(h[:, 0].min(), -0.5) and np.isclose(h[:, 0].max(), 4.5)
    assert np.isclose(h[:, 1].min(), -0.5) and np.isclose(h[:, 1].max(), 4.5)


def test_orient_array_puts_row_zero_on_top():
    xy = np.array([[0.0, 0.0], [1.0, 0.0]])            # (i, j): row 0 and row 1
    out = orient(xy, 'array')
    assert out[0, 1] > out[1, 1]


def test_edge_segments_torus_wrap_modes():
    lat = torus(4)                                     # 32 edges, 8 of them wrap around
    xy = np.asarray(lat.pos, dtype=float)
    segs, owner = edge_segments(xy, lat.edges, 'stubs')
    assert len(segs) == 24 + 2 * 8 and set(owner) == set(range(lat.n_edges))
    assert len(edge_segments(xy, lat.edges, 'hide')[0]) == 24
    assert len(edge_segments(xy, lat.edges, 'straight')[0]) == 32
    with pytest.raises(ValueError):
        edge_segments(xy, lat.edges, 'nope')


# ---------- palettes / frames ----------

def test_palette_lookup_and_fallback():
    rgba = ISING.rgba(np.array([-1, 1, 99, FREE]))
    assert rgba.shape == (4, 4)
    assert np.allclose(rgba[0], ISING.rgba(np.array([-1]))[0])
    assert np.allclose(rgba[2], ISING.fallback) and np.allclose(rgba[3], ISING.fallback)
    assert not np.allclose(rgba[0], rgba[1])


def test_default_palette_guess_and_categorical():
    assert default_palette(values=np.array([-1, 1])) is ISING
    assert len(categorical_palette(12).colours) == 12
    assert len(default_palette(values=np.array([0, 1, 2, 3])).colours) == 4


def test_ghost_values_from_boundary():
    lat = grid(3, ghosts=True)
    gv = ghost_values_from_boundary(lat, {lat.n_sites: 1})
    assert gv[0] == 1 and (gv[1:] == FREE).all()
    with pytest.raises(ValueError):
        ghost_values_from_boundary(lat, {0: 1})          # a real site, not a ghost


def test_rule_aligned_matches_manual():
    lat = grid(3)
    v = np.array([1, 1, -1, 1, -1, -1, 1, 1, 1])
    on = rule_aligned(lat, v, None)
    for e, (a, b) in enumerate(lat.edges):
        assert on[e] == float(v[a] == v[b])


# ---------- drawing ----------

def test_draw_grid_with_boundary():
    lat = grid(5, ghosts=True)
    bd = boundary_values(lat, 1)
    v = np.random.default_rng(0).choice([-1, 1], lat.n_sites)
    art = draw(lat, v, boundary=bd, edge_rule='aligned')
    assert art.nodes.get_facecolor().shape == (lat.null, 4)
    alpha = art.edges.get_edgecolor()[:, 3]
    assert set(np.round(alpha, 2)) <= {1.0, 0.12}
    assert len(art.ax.patches) == 2                      # boundary band drawn


def test_draw_without_ghosts_has_no_band():
    assert len(draw(grid(4), np.ones(16, dtype=int)).ax.patches) == 0


@pytest.mark.parametrize('make', [lambda: triangular(5), lambda: triangular(5, ghosts=True), lambda: torus(5)])
def test_draw_other_lattices(make):
    lat = make()
    draw(lat, np.ones(lat.n_sites, dtype=int), edge_rule='aligned')


def test_rc_edge_states_and_potts_overlay():
    lat = grid(4, ghosts=True)
    rng = np.random.default_rng(1)
    es = rng.integers(0, 2, lat.n_edges)
    v = rng.integers(0, 3, lat.n_sites)
    art = draw(lat, v, edge_states=es, boundary=boundary_values(lat, 0), palette=categorical_palette(3))
    alpha = art.edges.get_edgecolor()[:, 3]
    assert (alpha == 1.0).sum() == es.sum()
    draw(lat, None, edge_states=es)                       # RC only, neutral vertices


def test_shape_errors():
    lat = grid(3)
    with pytest.raises(ValueError):
        draw(lat, np.ones(5))
    with pytest.raises(ValueError):
        draw(lat, np.ones((2, 9)))                        # draw() is single-state only
    with pytest.raises(ValueError):
        animate(lat, np.ones((3, 9)), highlight=[0, 1])


def test_missing_pos_raises():
    from dataclasses import replace
    with pytest.raises(ValueError):
        draw(replace(grid(3), pos=None), np.ones(9, dtype=int))


def test_animate_and_save_gif(tmp_path):
    lat = grid(4, ghosts=True)
    T = 4
    states = np.random.default_rng(2).choice([-1, 1], (T, lat.n_sites))
    anim = animate(lat, states, boundary=boundary_values(lat, -1),
                   edge_rule='aligned', highlight=np.arange(-1, T - 1))
    out = tmp_path / 'a.gif'
    save(anim, out, fps=4)
    assert out.exists() and out.stat().st_size > 0
    with pytest.raises(ValueError):
        save(anim, tmp_path / 'a.avi')


# ---------- trajectory (needs the real models; numpy-only ones) ----------

def test_forward_trajectory_ising_and_animate():
    from perfectSim.models.ising import Ising
    lat = grid(5)
    m = Ising(lat, beta=0.3)
    states, upd = forward_trajectory(m, steps=20, stride=2, seed=0)
    assert states.shape == (11, 25) and upd.shape == (11,) and upd[0] == -1
    assert set(np.unique(states)) <= {-1, 1}
    anim = animate(lat, states, model=m, highlight=upd)
    assert anim is not None


def test_bounding_trajectory_starts_all_unknown():
    from perfectSim.models.ising import IsingAntiFerromagnetic
    lat = grid(5)
    m = IsingAntiFerromagnetic(lat, beta=-0.3)
    states, _ = forward_trajectory(m, steps=10, seed=0)
    assert (states[0] == m.unknown).all()
    draw(lat, states[-1], model=m)
