'''
Matplotlib 3D renderer.

Same design as the 2D LatticeArtist: every matplotlib object is built once
(one Line3DCollection for all edges, one 3D scatter for all vertices, one
translucent Poly3DCollection for the boundary band). After that, update(frame)
only overwrites colour arrays, which keeps animation cheap.

The renderer only ever sees a Frame (state values as plain arrays); it never
looks at the model.
'''

from typing import Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection
from perfectSim.lattice.lattice import FREE, Lattice
from perfectSim.viz.frames import Frame
from perfectSim.viz.palettes import Palette
from .geometry import as_3d, edge_segments, orient, padded_hull_faces


class LatticeArtist3D:
    def __init__(
        self,
        lat: Lattice,
        palette: Palette,
        ax=None,
        *,
        orientation: str = 'array',
        node_size: float = 60,
        ghost_size: float = 18,
        ghost_alpha: float = 0.7,
        ring_size: float = 240,
        band: float = 0.5,
        show_band: bool = True,
        band_alpha: float = 0.08,
        wrap: str = 'stubs',
        edge_width: float = 1.5,
        edge_colour='black',
        off_alpha: float = 0.06,
        slab: Optional[Tuple[int, float, float]] = None,
        elev: Optional[float] = None,
        azim: Optional[float] = None,
        depthshade: bool = False,
        show_axes: bool = False,
        figsize=(7, 7),
    ):
        '''
        slab: (axis, lo, hi), keep only sites (and edges between kept sites) whose
              lat.pos[:, axis] lies in [lo, hi]. Cuts the cube open so you can see
              inside, e.g. slab=(0, 0, 2) shows layers i = 0..2.
        elev, azim: camera angles in degrees. Default: (22, -60) for 3D lattices,
              (60, -90) for planar (2D) lattices.
        depthshade: fade far nodes (matplotlib's depth cue). Off by default so
              colours stay faithful to the palette.
        '''
        if lat.pos is None:
            raise ValueError('lattice has no coordinates (lat.pos is None); '
                             'build it with a constructor that sets pos, or supply a layout')
        raw = np.asarray(lat.pos, dtype=float)
        if raw.ndim != 2 or raw.shape[1] not in (2, 3):
            raise NotImplementedError(f'only 2D/3D coordinates are supported (pos shape {raw.shape})')
        if raw.shape[0] != lat.null:
            raise ValueError(f'lat.pos needs one row per site including boundary ghosts '
                             f'({lat.null}), got {raw.shape[0]}')

        self.lat, self.palette, self.off_alpha = lat, palette, off_alpha
        self.ghost_alpha = ghost_alpha
        self.xyz = orient(raw, orientation)
        n, nb = lat.n_sites, lat.n_boundary

        planar = raw.shape[1] == 2 or np.ptp(self.xyz[:, 2]) == 0
        self.elev = elev if elev is not None else (60.0 if planar else 22.0)
        self.azim = azim if azim is not None else (-90.0 if planar else -60.0)

        if ax is None:
            ax = plt.figure(figsize=figsize).add_subplot(projection='3d')
        elif not hasattr(ax, 'get_zlim'):
            raise ValueError("ax must be a 3D axes: fig.add_subplot(projection='3d')")
        self.ax, self.fig = ax, ax.figure

        # ---- slab: which vertices / edges are drawn at all -----------------------------
        vis = np.ones(n + nb, dtype=bool)
        if slab is not None:
            axis, lo, hi = slab
            col = raw[:, axis] if axis < raw.shape[1] else np.zeros(len(raw))
            vis = (col >= lo) & (col <= hi)
        self._vis = vis
        self._n_vis_sites = int(vis[:n].sum())

        # ---- boundary band: translucent padded hull around everything -------------------
        self.band = None
        if nb and show_band:
            faces = padded_hull_faces(self.xyz[vis], band)
            self.band = Poly3DCollection(faces, facecolor='0.6', edgecolor='none', alpha=band_alpha)
            self.ax.add_collection3d(self.band)

        # ---- edges: one collection; `owner` maps each segment back to its edge id -------
        segs, owner = edge_segments(self.xyz, lat.edges if lat.edges is not None else [], wrap=wrap)
        if len(owner):
            e = np.asarray(lat.edges)
            ok = vis[e[owner, 0]] & vis[e[owner, 1]]
            segs, owner = segs[ok], owner[ok]
        self._owner = owner
        self._edge_rgb = np.array(to_rgba(edge_colour))
        self.edges = None
        if len(owner):
            self.edges = Line3DCollection(segs, linewidths=edge_width)
            self.ax.add_collection3d(self.edges)

        # ---- vertices: one scatter, real sites then ghosts (same order as lat.pos) ------
        sizes = np.r_[np.full(n, node_size), np.full(nb, ghost_size)][vis]
        p = self.xyz[vis]
        self.nodes = self.ax.scatter(p[:, 0], p[:, 1], p[:, 2], s=sizes, depthshade=depthshade,
                                     edgecolors='black', linewidths=0.6)

        # ---- highlight ring (e.g. the site updated this step) ---------------------------
        p0 = self.xyz[0]
        self.ring = self.ax.scatter([p0[0]], [p0[1]], [p0[2]], s=ring_size, facecolors='none',
                                    edgecolors='crimson', linewidths=2.2, depthshade=False)
        self.ring.set_visible(False)

        # ---- view -----------------------------------------------------------------------
        lo_, hi_ = self.xyz[vis].min(axis=0), self.xyz[vis].max(axis=0)
        pad = band if (nb and show_band) else 0.3
        lo_, hi_ = lo_ - pad, hi_ + pad
        span = np.maximum(hi_ - lo_, 1e-6)
        self.ax.set_xlim(lo_[0], hi_[0])
        self.ax.set_ylim(lo_[1], hi_[1])
        self.ax.set_zlim(lo_[2], hi_[2])
        self.ax.set_box_aspect(span, zoom=1.2)  # true (equal-unit) aspect ratio
        self.ax.view_init(self.elev, self.azim)
        if not show_axes:
            self.ax.set_axis_off()

    # ------------------------------------------------------------------------------------

    def set_view(self, elev: Optional[float] = None, azim: Optional[float] = None):
        '''Move the camera (used by animate(..., spin=...)).'''
        self.elev = self.elev if elev is None else elev
        self.azim = self.azim if azim is None else azim
        self.ax.view_init(self.elev, self.azim)

    @staticmethod
    def _move(coll, p):
        x, y, z = (np.array([c]) for c in p)
        if hasattr(coll, 'set_data_3d'):
            coll.set_data_3d(x, y, z)
        else:
            coll._offsets3d = (x, y, z)

    def update(self, frame: Frame, highlight: Optional[int] = None, title: Optional[str] = None):
        n, nb = self.lat.n_sites, self.lat.n_boundary

        fc = np.empty((n + nb, 4))
        fc[:n] = self.palette.rgba(frame.values) if frame.values is not None else 1.0
        if nb:
            gv = frame.ghost_values if frame.ghost_values is not None else np.full(nb, FREE)
            fc[n:] = self.palette.rgba(gv)
            fc[n:, 3] *= self.ghost_alpha
        self.nodes.set_facecolor(fc[self._vis])

        if self.edges is not None:
            alpha = self.off_alpha + (1.0 - self.off_alpha) * frame.edge_on
            ec = np.tile(self._edge_rgb, (len(self._owner), 1))
            ec[:, 3] = alpha[self._owner]
            self.edges.set_color(ec)

        if highlight is None or highlight < 0 or not self._vis[highlight]:
            self.ring.set_visible(False)
        else:
            self._move(self.ring, self.xyz[highlight])
            self.ring.set_visible(True)

        if title is not None:
            self.ax.set_title(title)
        return tuple(a for a in (self.nodes, self.edges, self.ring) if a is not None)
