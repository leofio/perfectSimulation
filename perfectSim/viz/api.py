'''
User-facing functions: draw (one state), animate (many states), save.
'''

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
from .artist import LatticeArtist
from .frames import build_frame, ghost_values_from_boundary, resolve_rule
from .palettes import default_palette


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


def draw(lat, values=None, *, boundary=None, edge_states=None, edge_rule='solid',
         palette=None, model=None, highlight=None, title=None, ax=None, **artist_kw):
    '''
    Draw a single state.

    values:      (n_sites,) vertex values, or None
    boundary:    {global ghost id: value}, the same dict you give the model
    edge_states: (n_edges,) edge strengths in [0,1] (e.g. random-cluster open/closed)
    edge_rule:   'solid' | 'aligned' | callable(lat, values, ghost_values) -> (n_edges,)
                 (ignored if edge_states is given)
    palette:     Palette; default is inferred from `model` or the values
    highlight:   site index to ring (e.g. the site just updated)
    artist_kw:   forwarded to LatticeArtist (node_size, orientation, wrap, band, ...)

    Returns the LatticeArtist (its .fig / .ax are available for saving or styling).
    '''
    vals, es, gv, rule, pal, T = _setup(lat, values, edge_states, boundary, edge_rule, palette, model)
    if T != 1:
        raise ValueError('draw() takes a single state; use animate() for a sequence '
                         'or pass values[t]')
    art = LatticeArtist(lat, pal, ax=ax, **artist_kw)
    art.update(_frame(lat, vals, es, gv, rule, 0), highlight=highlight, title=title)
    return art


def animate(lat, values=None, *, boundary=None, edge_states=None, edge_rule='solid',
            palette=None, model=None, highlight=None, title='t = {t}', interval=80,
            repeat=True, **artist_kw):
    '''
    Animate a sequence of states. Arguments as draw(), except:

    values / edge_states: (T, n_sites) / (T, n_edges), one row per frame
    highlight:            (T,) site index per frame (-1 = none), or None
    title:                format string with {t}, or None
    interval:             milliseconds per frame

    Returns a FuncAnimation. Keep a reference to it, then show it or save(anim, 'x.gif').
    '''
    vals, es, gv, rule, pal, T = _setup(lat, values, edge_states, boundary, edge_rule, palette, model)
    hl = None
    if highlight is not None:
        hl = np.asarray(highlight)
        if hl.shape != (T,):
            raise ValueError(f'highlight must have shape ({T},), got {hl.shape}')

    art = LatticeArtist(lat, pal, **artist_kw)

    def step(t):
        return art.update(_frame(lat, vals, es, gv, rule, t),
                          highlight=None if hl is None else int(hl[t]),
                          title=title.format(t=t) if title else None)

    return FuncAnimation(art.fig, step, frames=T, interval=interval, blit=False, repeat=repeat)


def save(anim, path, fps: int = 12, dpi: int = 100):
    '''Write an animation to .gif (Pillow) or .mp4 (needs ffmpeg installed).'''
    ext = Path(path).suffix.lower()
    if ext == '.gif':
        writer = PillowWriter(fps=fps)
    elif ext == '.mp4':
        writer = FFMpegWriter(fps=fps)
    else:
        raise ValueError(f'unsupported extension {ext!r}; use .gif or .mp4')
    anim.save(str(path), writer=writer, dpi=dpi)

def animate_many(specs, *, interval=80, repeat=True, figsize=None, suptitle=None):
    '''
    Animate several lattices side by side on ONE shared timeline: a single
    FuncAnimation drives every panel from the same frame index, so they step in
    lock-step (two separately-created animations would each run on their own
    timer and drift apart).
 
    specs: one dict per panel, using the same keyword arguments as animate():
        {'lat': ..., 'values': ..., 'boundary': ..., 'edge_states': ...,
         'edge_rule': ..., 'palette': ..., 'model': ..., 'highlight': ...,
         'title': ..., **artist_kw}
        Only 'lat' is required. artist_kw (node_size, orientation, wrap, band, ...)
        is forwarded to that panel's LatticeArtist.
 
    Panels may have different numbers of frames; once a shorter panel runs out
    it just holds on its last frame while the others keep going. `title` may
    use {t}, which is filled with that panel's own (clamped) frame index.
 
    Returns a FuncAnimation over max(T_i) frames.
    '''
    if not specs:
        raise ValueError('animate_many needs at least one spec')
 
    fig, axes = plt.subplots(1, len(specs), figsize=figsize or (5.5 * len(specs), 5.5))
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
        # whatever's left in spec is forwarded straight to LatticeArtist
 
        vals, es, gv, rule, pal, T = _setup(lat, values, edge_states, boundary, edge_rule, palette, model)
        hl = None
        if highlight is not None:
            hl = np.asarray(highlight)
            if hl.shape != (T,):
                raise ValueError(f'highlight must have shape ({T},), got {hl.shape}')
 
        art = LatticeArtist(lat, pal, ax=ax, **spec)
        panels.append((art, lat, vals, es, gv, rule, hl, title, T))
        T_max = max(T_max, T)
 
    def step(t):
        drawn = []
        for art, lat, vals, es, gv, rule, hl, title, T in panels:
            tt = min(t, T - 1)
            frame = _frame(lat, vals, es, gv, rule, tt)
            h = None if hl is None else int(hl[tt])
            drawn.extend(art.update(frame, highlight=h, title=title.format(t=tt) if title else None))
        return drawn
 
    return FuncAnimation(fig, step, frames=T_max, interval=interval, blit=False, repeat=repeat)
 
 
def animate_pair(lat_a, values_a, lat_b, values_b, *, title_a=None, title_b=None, **kw):
    '''
    Convenience wrapper for the common two-panel case: animate_many with exactly
    two panels. lat_a/values_a and lat_b/values_b can be entirely different
    lattices and models (e.g. two boundary conditions, or a top-started vs
    bottom-started CFTP chain on the same lattice).
 
    kw is split between the two panels by suffix: pass boundary_a=/boundary_b=,
    model_a=/model_b=, edge_rule_a=/edge_rule_b=, etc. Anything passed without
    a suffix (interval=, figsize=, ...) goes to animate_many itself.
    '''
    panel_keys = {'boundary', 'edge_states', 'edge_rule', 'palette', 'model', 'highlight'}
    spec_a, spec_b, shared = {'lat': lat_a, 'values': values_a, 'title': title_a}, \
                             {'lat': lat_b, 'values': values_b, 'title': title_b}, {}
    for key, val in kw.items():
        if key.endswith('_a') and key[:-2] in panel_keys:
            spec_a[key[:-2]] = val
        elif key.endswith('_b') and key[:-2] in panel_keys:
            spec_b[key[:-2]] = val
        else:
            shared[key] = val
    return animate_many([spec_a, spec_b], **shared)
 