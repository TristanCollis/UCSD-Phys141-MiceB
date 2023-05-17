from typing import Any

import numpy as np

from src import gravity


def generate_points(n: int=10) -> tuple[np.ndarray[float, Any], np.ndarray[float, Any], np.ndarray[float, Any]]:
    massless = np.linspace(-10, 10, n*3).reshape(-1,3)
    massive = np.linspace(-5, 5, 2*3).reshape(-1,3)
    masses = np.ones(2)

    return massless, massive, masses


def test_3body_valid():
    gravity.grav_3body(*generate_points(), epsilon=1)


def test_2body_valid():
    gravity.grav_direct(*generate_points()[1:], epsilon=1)
