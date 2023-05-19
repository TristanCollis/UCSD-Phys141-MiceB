from typing import Any

import numpy as np

from src import gravnb
from src import gravity
from .utils import generate_points #type: ignore


def test_3body_valid():
    gravnb.grav_3body(*generate_points(), epsilon=1)


def test_3body_accurate():
    points = generate_points()
    deviation = np.linalg.norm(gravnb.grav_3body(*points, epsilon=1) - gravity.grav_3body(*points, epsilon=1))
    assert deviation < .001


def test_direct_valid():
    gravnb.grav_direct(*generate_points()[1:], epsilon=1)


def test_direct_accurate():
    points = generate_points()
    assert (gravnb.grav_direct(*points[1:], epsilon=1) == gravity.grav_direct(*points[1:], epsilon=1)).all()