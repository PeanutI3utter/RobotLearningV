# Learns the weights for the basis functions.
#
# Q_IM, QD_IM, QDD_IM are vectors containing positions, velocities and
# accelerations of the two joints obtained from the trajectory that we want
# to imitate.
#
# DT is the time step.
#
# NSTEPS are the total number of steps.

from getDMPBasis import *
import numpy as np

class dmpParams():
    def __init__(self):
        self.alphaz = 0.0
        self.alpha = 0.0
        self.beta = 0.0
        self.Ts = 0.0
        self.tau = 0.0
        self.nBasis = 0.0
        self.goal = 0.0
        self.w = 0.0

def dmpTrain (q, qd, qdd, dt, nSteps):

    params = dmpParams()
    #Set dynamic system parameters
    params.alphaz =
    params.alpha  =
    params.beta	 =
    params.Ts     =
    params.tau    =
    params.nBasis =
    params.goal   =

    Phi = getDMPBasis(params, dt, nSteps)

    #Compute the forcing function
    ft =

    #Learn the weights
    params.w =

    return params
