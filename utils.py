""" A collections of useful standalone functions
"""

import numpy as np
import matplotlib.pyplot as plt
from mytypes import Matrix, Vector
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def merge_arrays(*arrays: Vector) -> Vector:
    """Merge multiple one-dimensional arrays into a single array.

    Parameters:
    ----------
      *arrays: Multiple one-dimensional NumPy arrays.

    Returns:
      Merged one-dimensional NumPy array.
    """
    merged_array = np.concatenate(arrays)
    return merged_array

def _set_axes_radius(ax, origin, radius):
    x, y, z = origin
    ax.set_xlim3d([x - radius, x + radius])
    ax.set_ylim3d([y - radius, y + radius])
    ax.set_zlim3d([z - radius, z + radius])
        
def set_axes_equal(ax: plt.Axes) -> None:
    """This function is used to make the 
    3-side measurements of the 3D image consistent
    """
    limits = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()])
    origin = np.mean( limits, axis=1 )
    radius = 0.5*np.max(np.abs(limits[:, 1] - limits[:, 0]))
    _set_axes_radius(ax, origin, radius)

def get_norm_direction(dir: Vector) -> (Vector, Vector):
    """This function is used to get the orthogonal
    vector and normalize them

    Paramters
    ---------
    dir:
      2d given direction
    
    Returns
    -------
    normalized direction and the orthogonal direction
    """
    dir_norm = np.linalg.norm(dir, ord=2)
    dir_y = dir/dir_norm
    dir_x = np.array([dir_y[1], -dir_y[0]])
    return dir_x, dir_y

def get_one_corner(sign: tuple, center: Vector, dir_x: Vector, dir_y: Vector, 
                   half_length: float, half_width: float) -> Vector:
    """This function will give the coordinates of one corner
    """
    v = center[0:2] + sign[0]*half_width*dir_x + sign[1]*half_length*dir_y
    v = np.append(v, center[2])
    return v

def get_corners(table: tuple) -> list:
    """This function will calculate the coordinates
    of four corners of the table
    """
    center = table[0]
    dir = table[1]
    half_length = table[2]
    half_width = table[3]

    dir_x, dir_y = get_norm_direction(dir)
    sign_list = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
    v = []
    for i in range(len(sign_list)):
        v.append(get_one_corner(sign_list[i], center, dir_x, dir_y, half_length, half_width))
    return v


def add_table(ax: plt.Axes, table: tuple) -> None:
    """This function will plot a table with 
    given parameters
    """
    v = get_corners(table)
    vtx = list([tuple(v[0]), tuple(v[2]), tuple(v[3]) , tuple(v[1])])
    area = Poly3DCollection([vtx], facecolors='lightsteelblue', linewidths=1, alpha=0.5) 
    area.set_edgecolor('k')
    ax.add_collection3d(area)

def plot_trajectories(*trajectories: Matrix, table: tuple) -> None:
    """ Plots 3d trajectories 

    Parameters
    ----------
    trajectories
      a list of list of 3d positions
    
    """
    ax = plt.axes(projection="3d")
    if len(table) > 0:
        add_table(ax, table)
    for trajectory in trajectories:
        assert trajectory.shape[1] >= 3
        ax.scatter3D(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], s=5)
    set_axes_equal(ax)
    plt.show()
