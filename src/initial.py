from typing import Any

import numpy as np
import matplotlib.pyplot as plt

#The following functions assume the plane of orbit is the x, y axis, with positive 

def initmice(R_min:float, mass:float, time_unit:float, wA:float, wb:float, ia:float, ib:float, e:float, epsilon:float) -> tuple :

    #Disk Inits in galaxy frame
    posA, velA = initdisk(R_min, mass, epsilon)
    posB, velB = initdisk(R_min, mass, epsilon)

    Rapoc = R_min*(1+e)/(1-e)

    #Transform into galaxy frame using angles TODO
    velA_CM = velA
    velB_CM = velB


    #Translate each galacy so COM is at origin
    pos_m = np.array([[0, Rapoc, 0], [0, -Rapoc, 0]])

    posA_CM = posA+pos_m[0]
    posB_CM = posB+pos_m[1]

    pos_CM = np.concatenate((posA_CM, posB_CM), axis=0)
    vel_CM = np.concatenate((velA_CM, velB_CM), axis=0)

    return pos_CM, pos_CM, vel_CM

    
def initdisk(R_min:float, mass:float, epsilon:float) ->tuple:
    pos = np.zeros((297, 3))
    vel = np.zeros((297, 3))
    curr = 0
    for i in range(0, 11):
        num_particles = 3*i + 12
        radius = (0.05*i + 0.2)*R_min
        for j in range(0, num_particles):
            angle = (2.0*np.pi /num_particles)*j
            pos[curr] = (radius*np.cos(angle), radius*np.sin(angle), 0)
            velmag = np.sqrt(mass*radius/(radius**2+epsilon**2))
            veldirnorm = np.array([-1*pos[curr, 1], pos[curr, 0], 0]) / np.sqrt(pos[curr, 1]**2+pos[curr, 0]**2)
            vel[curr] = velmag*veldirnorm
            curr+=1
    return pos, vel
