'''
Matplotlib renderer.

LatticeArtist builds every matplotlib object once (one LineCollection for all
edges, one scatter for all vertices, polygons for the boundary band). After
that, update(frame) only overwrites colour arrays, which is what makes
animation cheap.
'''

from typing import Optional
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import to_rgba
from matplotlib.patches import Polygon
from perfectSim.lattice.lattice import FREE, Lattice
from .frames import Frame
from .geometry import edge_segments, orient, padded_hull
from .palettes import Palette


class LatticeArtist:
    def __init__(
        self,
        lat: Lattice,
        palette: Palette,
        ax=None,
        *,
        orientation: str = 'array',
        node_size: float = 90,
        ghost_size: float = 35,
        ring_size: float = 280,
        band: float = 0.5,
        show_band: bool = True,
        wrap: str = 'stubs',
        edge_width: float = 2.0,
        edge_colour='black',
        off_alpha: float = 0.12,
        figsize=(6, 6),
    ):
        if lat.pos is None:
            raise ValueError('lattice has no coordinates (lat.pos is None); '
                             'build it with a constructor that sets pos, or supply a layout')
        pos = np.asarray(lat.pos, dtype=float)
        if pos.ndim != 2 or pos.shape[1] != 2:
            raise NotImplementedError(f'only 2D lattices are supported so far (pos shape {pos.shape})')
        if pos.shape[0] != lat.null:
            raise ValueError(f'lat.pos needs one row per site including boundary ghosts '
                             f'({lat.null}), got {pos.shape[0]}')

        self.lat, self.palette, self.off_alpha = lat, palette, off_alpha
        self.xy = orient(pos, orientation)
        self.ax = ax if ax is not None else plt.subplots(figsize=figsize)[1]
        self.fig = self.ax.figure
        n, nb = lat.n_sites, lat.n_boundary

        # Boundary band: grey hull around everything, white hull around the interior.
        if nb and show_band:
            self.ax.add_patch(Polygon(padded_hull(self.xy, band), closed=True,
                                      fc='0.85', ec='none', zorder=0))
            self.ax.add_patch(Polygon(padded_hull(self.xy[:n], band), closed=True,
                                      fc='white', ec='none', zorder=0.5))

        # Edges: one collection; `owner` maps each drawn segment back to its edge id.
        segs, self._owner = edge_segments(self.xy, lat.edges, wrap=wrap)
        self._edge_rgb = np.array(to_rgba(edge_colour))
        self.edges = LineCollection(segs, linewidths=edge_width, zorder=1)
        self.ax.add_collection(self.edges)

        # Vertices: one scatter, real sites then ghosts (same ordering as lat.pos).
        sizes = np.r_[np.full(n, node_size), np.full(nb, ghost_size)]
        self.nodes = self.ax.scatter(self.xy[:, 0], self.xy[:, 1], s=sizes, zorder=2,
                                     edgecolors='black', linewidths=0.8)

        # Optional highlight ring (e.g. the site being updated this step).
        self.ring = self.ax.scatter([], [], s=ring_size, facecolors='none',
                                    edgecolors='crimson', linewidths=2.2, zorder=3)

        self.ax.set_aspect('equal')
        self.ax.autoscale_view()
        self.ax.margins(0.03)
        self.ax.axis('off')

    def update(self, frame: Frame, highlight: Optional[int] = None, title: Optional[str] = None):
        n, nb = self.lat.n_sites, self.lat.n_boundary

        fc = np.empty((n + nb, 4))
        fc[:n] = self.palette.rgba(frame.values) if frame.values is not None else 1.0
        if nb:
            gv = frame.ghost_values if frame.ghost_values is not None else np.full(nb, FREE)
            fc[n:] = self.palette.rgba(gv)
        self.nodes.set_facecolor(fc)

        alpha = self.off_alpha + (1.0 - self.off_alpha) * frame.edge_on
        ec = np.tile(self._edge_rgb, (len(self._owner), 1))
        ec[:, 3] = alpha[self._owner]
        self.edges.set_color(ec)

        if highlight is None or highlight < 0:
            self.ring.set_offsets(np.empty((0, 2)))
        else:
            self.ring.set_offsets(self.xy[[highlight]])

        if title is not None:
            self.ax.set_title(title)
        return self.nodes, self.edges, self.ring
