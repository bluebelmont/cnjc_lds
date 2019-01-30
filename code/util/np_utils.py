""" Utility for common numpy operations
"""

__author__ = 'Blue Sheffer'

import numpy as np
import numpy.linalg as LA


def random_rotation(n, theta=None):
    """
    Args:
        n (int): number of rows/columns for rotation matrix
        theta (float): angle of rotation

    Returns:
        rotation_matrix (np.ndarray): rotation matrix of shape (n, n)

    """
    if theta is None:
        # Sample a random, slow rotation
        theta = 0.5 * np.pi * np.random.rand()

    if n == 1:
        return np.random.rand() * np.eye(1)

    rot = np.array([[np.cos(theta), -np.sin(theta)],
                    [np.sin(theta), np.cos(theta)]])
    out = np.zeros((n, n))
    out[:2, :2] = rot
    q = np.linalg.qr(np.random.randn(n, n))[0]
    rotation_matrix = q.dot(out).dot(q.T)
    return rotation_matrix


def sort_by_column_norm(A, *args):
    descending_column_order = np.argsort(LA.norm(A, axis=0))[::-1]
    sorted_args = []
    for arg in args:
        sorted_args.append(arg[:,descending_column_order])
    if len(sorted_args) > 0:
        return A[:, descending_column_order], sorted_args
    else:
        return A[:, descending_column_order]
