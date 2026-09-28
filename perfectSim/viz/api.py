'''
User-facing functions: draw (one state), animate (many states), save.
'''

from pathlib import Path
import numpy as np
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
