from typing import Any

import numpy as np

rng = np.random.default_rng()


def generate_points(n: int = 10) -> tuple[np.ndarray[float, Any], np.ndarray[float, Any], np.ndarray[float, Any]]:
    massless = np.linspace(-10, 10, n*3).reshape(-1, 3)
    massive = np.linspace(-5, 5, 2*3).reshape(-1, 3)
    masses = np.ones(2)

    return massless, massive, masses


def rng_points(n: int = 10) -> tuple[np.ndarray[float, Any], np.ndarray[float, Any], np.ndarray[float, Any]]:
    massless = rng.random(n*3).reshape(-1, 3) * 20 - 10
    massive = rng.random(2 * 3).reshape(-1, 3) * 10 - 5
    masses = np.ones(2)

    return massless, massive, masses
