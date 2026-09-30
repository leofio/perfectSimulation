'''
User-facing functions for 3D: draw (one state), animate (many states), save,
animate_many / animate_pair (lock-step panels).

Signatures mirror perfectSim.viz.api. Everything is driven by state arrays:
    values       (n_sites,)  or (T, n_sites)   vertex states
    edge_states  (n_edges,)  or (T, n_edges)   edge strengths in [0, 1]
The renderer never runs a model; it only draws the arrays you hand it.
'''

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from perfectSim.viz.api import save  # noqa: F401  (gif/mp4 writer, dimension-agnostic)
from perfectSim.viz.frames import build_frame, ghost_values_from_boundary, resolve_rule
from perfectSim.viz.palettes import default_palette
from .artist import LatticeArtist3D


def _as_stack(a, width, name):
    '''None -> None; (width,) -> (1, width); (T, width) -> unchanged.'''
    if a is None:
        return None
    a = np.asarray(a)
    if a.ndim == 1:
        a = a[None, :]
    if a.ndim != 2 or a.shape[1] != width:
        raise ValueError(f'{name} must have shape ({width},) or (T, {width}); got {a.shape}')
    return a


def _setup(lat, values, edge_states, boundary, edge_rule, palette, model):
    vals = _as_stack(values, lat.n_sites, 'values')
    es = _as_stack(edge_states, lat.n_edges, 'edge_states')
    if vals is not None and es is not None and len(vals) != len(es):
        raise ValueError(f'values has {len(vals)} frames but edge_states has {len(es)}')
    T = len(vals) if vals is not None else (len(es) if es is not None else 1)
    gv = ghost_values_from_boundary(lat, boundary) if lat.n_boundary else None
    pal = palette if palette is not None else default_palette(model, vals)
    return vals, es, gv, resolve_rule(edge_rule), pal, T


def _frame(lat, vals, es, gv, rule, t):
    return build_frame(lat,
                       None if vals is None else vals[t],
                       gv,
                       None if es is None else es[t],
                       rule)


def _check_highlight(highlight, T):
    if highlight is None:
        return None
    hl = np.asarray(highlight)
    if hl.shape != (T,):
        raise ValueError(f'highlight must have shape ({T},), got {hl.shape}')
    return hl


def draw(lat, values=None, *, boundary=None, edge_states=None, edge_rule='solid',
         palette=None, model=None, highlight=None, title=None, ax=None, **artist_kw):
    '''
    Draw a single state in 3D.

    values:      (n_sites,) vertex values, or None
    boundary:    {global ghost id: value}, the same dict you give the model
    edge_states: (n_edges,) edge strengths in [0,1] (e.g. random-cluster open/closed)
    edge_rule:   'solid' | 'aligned' | callable(lat, values, ghost_values) -> (n_edges,)
                 (ignored if edge_states is given)
    palette:     Palette; default is inferred from `model` or the values
    highlight:   site index to ring (e.g. the site just updated)
    ax:          an existing 3D axes (fig.add_subplot(projection='3d'))
    artist_kw:   forwarded to LatticeArtist3D (node_size, orientation, wrap, band,
                 slab, elev, azim, off_alpha, depthshade, ...)

    Returns the LatticeArtist3D (.fig / .ax available for saving or styling).
    '''
    vals, es, gv, rule, pal, T = _setup(lat, values, edge_states, boundary, edge_rule, palette, model)
    if T != 1:
        raise ValueError('draw() takes a single state; use animate() for a sequence '
                         'or pass values[t]')
    art = LatticeArtist3D(lat, pal, ax=ax, **artist_kw)
    art.update(_frame(lat, vals, es, gv, rule, 0), highlight=highlight, title=title)
    return art


def animate(lat, values=None, *, boundary=None, edge_states=None, edge_rule='solid',
            palette=None, model=None, highlight=None, title='t = {t}', interval=80,
            repeat=True, spin=0.0, **artist_kw):
    '''
    Animate a sequence of states in 3D. Arguments as draw(), except:

    values / edge_states: (T, n_sites) / (T, n_edges), one row per frame
    highlight:            (T,) site index per frame (-1 = none), or None
    title:                format string with {t}, or None
    interval:             milliseconds per frame
    spin:                 camera azimuth change in degrees per frame (0 = fixed camera)

    Returns a FuncAnimation. Keep a reference to it, then show it or save(anim, 'x.gif').
    '''
    vals, es, gv, rule, pal, T = _setup(lat, values, edge_states, boundary, edge_rule, palette, model)
    hl = _check_highlight(highlight, T)
    art = LatticeArtist3D(lat, pal, **artist_kw)
    az0 = art.azim

    def step(t):
        if spin:
            art.set_view(azim=az0 + spin * t)
        return art.update(_frame(lat, vals, es, gv, rule, t),
                          highlight=None if hl is None else int(hl[t]),
                          title=title.format(t=t) if title else None)

    return FuncAnimation(art.fig, step, frames=T, interval=interval, blit=False, repeat=repeat)


def animate_many(specs, *, interval=80, repeat=True, figsize=None, suptitle=None):
    '''
    Animate several 3D lattices side by side on ONE shared timeline (a single
    FuncAnimation drives every panel, so they step in lock-step).

    specs: one dict per panel, same keywords as animate():
        {'lat': ..., 'values': ..., 'boundary': ..., 'edge_states': ...,
         'edge_rule': ..., 'palette': ..., 'model': ..., 'highlight': ...,
         'title': ..., 'spin': ..., **artist_kw}
        Only 'lat' is required.

    Panels may have different numbers of frames; a shorter panel holds on its
    last frame. `title` may use {t} (that panel's own clamped frame index).

    Returns a FuncAnimation over max(T_i) frames.
    '''
    if not specs:
        raise ValueError('animate_many needs at least one spec')

    fig, axes = plt.subplots(1, len(specs), figsize=figsize or (6 * len(specs), 6),
                             subplot_kw={'projection': '3d'})
    axes = np.atleast_1d(axes)
    if suptitle:
        fig.suptitle(suptitle)

    panels, T_max = [], 0
    for spec, ax in zip(specs, axes):
        spec = dict(spec)
        lat = spec.pop('lat')
        values = spec.pop('values', None)
        boundary = spec.pop('boundary', None)
        edge_states = spec.pop('edge_states', None)
        edge_rule = spec.pop('edge_rule', 'solid')
        palette = spec.pop('palette', None)
        model = spec.pop('model', None)
        highlight = spec.pop('highlight', None)
        title = spec.pop('title', None)
        spin = spec.pop('spin', 0.0)

        vals, es, gv, rule, pal, T = _setup(lat, values, edge_states, boundary, edge_rule, palette, model)
        hl = _check_highlight(highlight, T)
        art = LatticeArtist3D(lat, pal, ax=ax, **spec)
        panels.append((art, lat, vals, es, gv, rule, hl, title, spin, art.azim, T))
        T_max = max(T_max, T)

    def step(t):
        drawn = []
        for art, lat, vals, es, gv, rule, hl, title, spin, az0, T in panels:
            tt = min(t, T - 1)
            if spin:
                art.set_view(azim=az0 + spin * t)
            h = None if hl is None else int(hl[tt])
            drawn.extend(art.update(_frame(lat, vals, es, gv, rule, tt), highlight=h,
                                    title=title.format(t=tt) if title else None))
        return drawn

    return FuncAnimation(fig, step, frames=T_max, interval=interval, blit=False, repeat=repeat)


def animate_pair(lat_a, values_a, lat_b, values_b, *, title_a=None, title_b=None, **kw):
    '''
    Two-panel convenience wrapper around animate_many (e.g. top- vs bottom-started
    CFTP chains). Per-panel keywords take an _a / _b suffix (boundary_a=, model_b=,
    edge_rule_a=, ...); anything unsuffixed (interval=, figsize=, ...) goes to animate_many.
    '''
    panel_keys = {'boundary', 'edge_states', 'edge_rule', 'palette', 'model', 'highlight', 'spin'}
    spec_a = {'lat': lat_a, 'values': values_a, 'title': title_a}
    spec_b = {'lat': lat_b, 'values': values_b, 'title': title_b}
    shared = {}
    for key, val in kw.items():
        if key.endswith('_a') and key[:-2] in panel_keys:
            spec_a[key[:-2]] = val
        elif key.endswith('_b') and key[:-2] in panel_keys:
            spec_b[key[:-2]] = val
        else:
            shared[key] = val
    return animate_many([spec_a, spec_b], **shared)
