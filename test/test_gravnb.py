import numpy as np

from src import gravity, gravnb

rng = np.random.default_rng()

def test_accuracy_3body():
    num_massless = 10
    num_massive = 2

    massless = rng.random((num_massless, 3))
    massive = rng.random((num_massive, 3))
    masses = np.ones(num_massive)
    epsilon = 0.1

    assert (gravity.grav_3body(massless, massive, masses, epsilon) == gravnb.grav_3body(massless, massive, masses, epsilon)).all()


def test_accuracy_2body():
    num_massive = 2

    massive = rng.random((num_massive, 3))
    masses = np.ones(num_massive)
    epsilon = 0.1

    assert (gravity.grav_direct(massive, masses, epsilon) == gravnb.grav_direct(massive, masses, epsilon)).all()