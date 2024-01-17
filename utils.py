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

def tuple2array():
    pass

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
    dir_norm = np.linalg.norm(dir, ord=2)
    dir_y = dir/dir_norm
    dir_x = np.array([dir_y[1], -dir_y[0]])
    return dir_x, dir_y

def get_corners(table: tuple) -> tuple:
    center = table[0]
    dir = table[1]
    half_length = table[2]
    half_width = table[3]

    dir_x, dir_y = get_norm_direction(dir)
    sign_list = [(1, 1), (1, -1), (-1, 1), (-1, -1)]

    v1 = center[0:2] + half_width*dir_x + half_length*dir_y
    v1 = np.append(v1, center[2])

    v2 = center[0:2] + half_width*dir_x - half_length*dir_y
    v2 = np.append(v2, center[2])

    v3 = center[0:2] - half_width*dir_x + half_length*dir_y
    v3 = np.append(v3, center[2])

    v4 = center[0:2] - half_width*dir_x - half_length*dir_y
    v4 = np.append(v4, center[2])
    return v1, v2, v3, v4


def add_table(ax: plt.Axes, table: tuple) -> None:
    v1, v2, v3, v4 = get_corners(table)
    vtx = list([tuple(v1), tuple(v3), tuple(v4) , tuple(v2)])
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
