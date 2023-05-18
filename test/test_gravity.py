from typing import Any

import numpy as np

from src import gravity
from .utils import generate_points #type: ignore


def test_3body_valid():
    gravity.grav_3body(*generate_points(), epsilon=1)


def test_2body_valid():
    gravity.grav_direct(*generate_points()[1:], epsilon=1)
