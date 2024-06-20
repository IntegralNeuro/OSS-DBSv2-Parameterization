import numpy as np
import pyvista as pv
from matplotlib import pyplot as plt

"""
Some simple tools for analyzing the simulation results.
"""


def load_vtu(vtu_file: str) -> tuple:
    """
    Load the vtu file and return the scalar values and the points.

    Args:
        vtu_file (str): The path to the vtu file.

    Returns:
        points: np.ndarray: The grid points, shape (n_points, 3)
        values: list of np.ndarray, each (n_points, dim)
        names: list of array names
    """
    mesh = pv.read(vtu_file)
    values = []
    for i, name in enumerate(mesh.array_names):
        values.append(mesh[mesh.array_names[i]])
    return mesh.points, values, mesh.array_names

