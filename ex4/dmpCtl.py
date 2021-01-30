# DMP-based controller.
#
# DMPPARAMS is the struct containing all the parameters of the DMP.
#
# PSI is the vector of basis functions.
#
# Q and QD are the current position and velocity, respectively.

import numpy as np

def dmpCtl (dmpParams, psi_i, q, qd):

    qdd = (
        dmpParams.tau**2 * (dmpParams.alpha * (dmpParams.beta *
            (dmpParams.goal - q) - qd / dmpParams.tau) + psi_i.T @ dmpParams.w)
    )

    return qdd

