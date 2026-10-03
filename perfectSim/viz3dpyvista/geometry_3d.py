'''
Convert Lattice to PyVista geometry.
'''

import numpy as np
import pyvista as pv
from perfectSim.lattice.lattice import Lattice

def lat_to_pyvista(lat: Lattice):
    '''Convert a Lattice object to PyVista PolyData mesh.'''
    pts = lat.pos

    if pts.shape[1] == 2:
        pts = np.column_stack((pts, np.zeros(len(pts))))

    if lat.n_edges > 0:
        padding = np.full((lat.n_edges,  1), 2, dtype=np.int32)
        lines = np.hstack([padding, lat.edges]).ravel()
    else:
        lines = None

    return pv.PolyData(pts, lines=lines)
