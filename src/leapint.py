import numpy as np
import gravity

def leapfrog(m, n, init_pos, init_vel, init_pos_massive, init_vel_massive, time, time_step, masses, epsilon):
    
    """leapfrog integration for 3 massless bodies

    Parameters
    ----------
    m : Scalar
        Number of Massive planet
    n:
        Number of Massless planet 
    init_pos : 
        Array of initial position of 3 massless bodies
    init_vel : 
        Array of initial velocity of 3 massless bodies
    init_pos_massive:
        Array of initial position of 2 massive points 
    init_vel_massive:
        Array of initial velocity of 2 massive points
    time: Scalar 
        number of time we integrate
    time_step: Scalar or float
        smaller time_step to do integration
    masses : np.ndarray[float, (2)]
        Masses corresponding to the given points.
    epsilon : float
        Softening factor

    Returns
    -------
    
    
    """
    #set the matrix that we will use in the leapfrog integrator
    pos = np.zeros((n,3))
    vel = np.zeros((n,3))
    pos_m = np.zeros((m,3))
    vel_m = np.zeros((m,3))

    
    # set the output array of 3 massless bodies 
    position = np.zeros((time/time_step +1,n,3))
    velocity = np.zeros((time/time_step +1,n,3))
    # set the first tensor of the output to be our initial condition
    position[0,:,:] = init_pos
    velocity[0,:,:] = init_vel
    
    # set the output array of 2 massive bodies 
    position_m = np.zeros((time/time_step +1,m,3))
    velocity_m = np.zeros((time/time_step +1,m,3))
    # set the first tensor of the output to be our initial condition
    position_m[0,:,:] = init_pos_massive
    velocity_m[0,:,:] = init_vel_massive
    
    for i in range(time/time_step):
        
        #find the half step velocity 
        vel = velocity[i,:,:] + (grav_3body(position[i,:,:],position_m[i,:,:],masses,epsilon)*time_step/2)
        vel_m = velocity_m[i,:,:] + (grav_2body(position_m[i,:,:],masses,epsilon)*time_step/2)
        
        #find the full-step position
        pos = position[i,:,:] + (vel*time_step)
        pos_m = position_m[i,:,:] + (vel_m*time_step)
        
        #find the full-step velocity
        vel = vel + (grav_3body(pos,pos_m,masses,epsilon)*time_step/2)
        vel_m = vel_m + (grav_2body(pos_m,masses,epsilon)*time_step/2)
        
        #add our integrated calculation to the output tensor
        position[i+1,:,:] = pos 
        velocity[i+1,:,:] = vel
        position_m[i+1,:,:] = pos_m 
        velocity_m[i+1,:,:] = vel_m
        
    return position, velocity, position_m, velocity_m
    
