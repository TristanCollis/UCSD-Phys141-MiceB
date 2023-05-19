from typing import Any

import numpy as np
from . import gravity
def leapfrog(init_pos: np.ndarray[float, Any], init_vel: np.ndarray[float, Any], init_pos_massive: np.ndarray[float, Any], init_vel_massive: np.ndarray[float, Any], timestep: int, dt: float, masses: np.ndarray[float, Any], epsilon: float) -> tuple:
    """leapfrog integration for 3 massless bodies

    Parameters
    ----------
    init_pos : np.ndarray[float, (n,3)]
        Array of initial position of 3 massless bodies
    init_vel : np.ndarray[float, (n,3)]
        Array of initial velocity of 3 massless bodies
    init_pos_massive : np.ndarray[float, (m,3)]
        Array of initial position of 2 massive points
    init_vel_massive : np.ndarray[float, (m,3)]
        Array of initial velocity of 2 massive points
    timestep : int
        number of time we integrate
    dt : float
        smaller time_step to do integration
    masses : np.ndarray[float, (m,)]
        Masses corresponding to the given points.
    epsilon : float
        Softening factor

    Returns
    -------
    tuple
        _description_
    """
    
    # set the number of massless particle as n and massive particle as m
    m,a = init_pos_massive.shape
    n,a = init_pos.shape
    
    # set the matrix that we will use in the leapfrog integrator
    pos = np.zeros((n, 3))
    vel = np.zeros((n, 3))
    pos_m = np.zeros((m, 3))
    vel_m = np.zeros((m, 3))

    # set the output array of 3 massless bodies
    position = np.zeros((timestep + 1, n, 3))
    velocity = np.zeros((timestep + 1, n, 3))

    # set the first tensor of the output to be our initial condition
    position[0, :, :] = init_pos
    velocity[0, :, :] = init_vel

    # set the output array of 2 massive bodies
    position_m = np.zeros((timestep + 1, m, 3))
    velocity_m = np.zeros((timestep + 1, m, 3))

    # set the first tensor of the output to be our initial condition
    position_m[0, :, :] = init_pos_massive
    velocity_m[0, :, :] = init_vel_massive

    for i in range(timestep):

        # find the half step velocity
        vel = velocity[i, :, :] + (gravity.grav_3body(position[i, :, :],
                                   position_m[i, :, :], masses, epsilon)*dt/2)
        vel_m = velocity_m[i, :, :] + \
            (gravity.grav_direct(
                position_m[i, :, :], masses, epsilon)*dt/2)

        # find the full-step position
        pos = position[i, :, :] + (vel*dt)
        pos_m = position_m[i, :, :] + (vel_m*dt)

        # find the full-step velocity
        vel = vel + (gravity.grav_3body(pos, pos_m,
                     masses, epsilon)*dt/2)
        vel_m = vel_m + (gravity.grav_direct(pos_m,
                         masses, epsilon)*dt/2)

        # add our integrated calculation to the output tensor
        position[i+1, :, :] = pos
        velocity[i+1, :, :] = vel
        position_m[i+1, :, :] = pos_m
        velocity_m[i+1, :, :] = vel_m

    return position, velocity, position_m, velocity_m
