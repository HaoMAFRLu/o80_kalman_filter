""" A collections of useful standalone functions
"""

import numpy as np
import matplotlib.pyplot as plt
from mytypes import Matrix, Vector

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