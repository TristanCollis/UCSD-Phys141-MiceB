from typing import Any

import numpy as np
from numba import njit, float64


@njit
def norm(array: np.ndarray[float, Any], axis: int) -> np.ndarray[float, Any]:
    return np.sqrt(np.sum(array**2, axis=axis))


@njit(float64[:,:](float64[:,:], float64[:,:], float64[:,:], float64))
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

    r = (np.expand_dims(massive_points, 1)
         - np.expand_dims(massless_points, 0))

    r_norm = np.expand_dims(norm(r, axis=2), 2)

    return np.sum(
        np.expand_dims(masses, (1, 1))
        * (r / r_norm)
        / (r_norm + epsilon)**2,
        axis=0
    )


@njit(float64[:,:](float64[:,:], float64[:,:], float64))
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


