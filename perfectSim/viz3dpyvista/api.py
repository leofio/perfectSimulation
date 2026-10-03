'''
User facing 3D viz.
'''

import numpy as np
import pyvista as pv
from perfectSim.viz3dpyvista.geometry_3d import lat_to_pyvista

def draw(lat, values):
    mesh = lat_to_pyvista(lat)

    plotter = pv.Plotter(off_screen=True)

    plotter.add_mesh(
        mesh,
        scalars=values,
        cmap='coolwarm',
        render_points_as_spheres=True,
        point_size=15,
        line_width=3,
        show_scalar_bar=False
    )
    return plotter

def animate(lat, states, filename='output_ani.gif'):
    mesh = lat_to_pyvista(lat)

    plotter = pv.Plotter(off_screen=True)

    mesh.point_data['State'] = states[0]
    plotter.add_mesh(
        mesh, 
        scalars="State", 
        cmap='viridis', 
        render_points_as_spheres=True, 
        point_size=15, 
        line_width=2,
        clim=[np.min(states), np.max(states)] 
    )
    
    
    if filename.endswith('.mp4'):
        plotter.open_movie(filename)
    else:
        plotter.open_gif(filename)
    
    for state in states[1:]:
        mesh.point_data["State"] = state
        plotter.write_frame()
        
    plotter.close()
    return plotter
