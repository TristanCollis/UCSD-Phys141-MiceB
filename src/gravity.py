from typing import Any

import numpy as np


def grav_3body(
        massless_points: np.ndarray[float, Any],
        massive_points: np.ndarray[float, Any],
        masses: np.ndarray[float, Any],
        epsilon: float,
) -> np.ndarray[float, Any]:
    """Restricted N-Body acceleration using a small number of massive bodies attracting a large number of massless bodies.

    Parameters
    ----------
    massless_points : np.ndarray[float, (m, 3)]
        Array of massless points.
    massive_points : np.ndarray[float, (n, 3)]
        Array of massive points.
    masses : np.ndarray[float, (n)]
        Masses of massive points.
    epsilon : float
        Softening factor

    Returns
    -------
    np.ndarray[float, (m, 3)]
        Accelerations to be applied to the massless bodies.
    """

    r = (massive_points[np.newaxis].transpose(1, 0, 2)
         - massless_points[np.newaxis])

    r_norm = np.linalg.norm(r, axis=-1)[np.newaxis].transpose(1, 2, 0)

    return np.sum(
        masses.reshape(-1, 1, 1)
        * r
        / (r_norm + epsilon)**3,
        axis=0
    )


def grav_direct(
        points: np.ndarray[float, Any],
        masses: np.ndarray[float, Any],
        epsilon: float,
) -> np.ndarray[float, Any]:
    """Direct gravity acceleration between n massive points.

    Parameters
    ----------
    points : np.ndarray[float, (n, 3)]
        Positions of the massive points.
    masses : np.ndarray[float, (n)]
        Masses corresponding to the given points.
    epsilon : float
        Softening factor

    Returns
    -------
    np.ndarray[float, (n, 3)]
        Acceleration on each massive point.
    """

    return grav_3body(points, points, masses, epsilon)
