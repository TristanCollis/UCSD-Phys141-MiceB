import numpy as np
from src import leapint

from .utils import generate_points

def test_step():
    massless, massive, masses = generate_points()

    leapint.leapfrog(massless, np.zeros_like(massless), massive, np.zeros_like(massive), 1, 1, masses, 0.1)


def test_integration():
    massless, massive, masses = generate_points()

    leapint.leapfrog(massless, np.zeros_like(massless), massive, np.zeros_like(massive), 100, 1, masses, 0.1)