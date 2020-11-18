# CTL is the name of the controller.
# Q_HISTORY is a matrix containing all the past position of the robot. Each row of this matrix is [q_1, ... q_i], where
# i is the number of the joints.
# Q and QD are the current position and velocity, respectively.
# Q_DES, QD_DES, QDD_DES are the desired position, velocity and acceleration, respectively.
# GRAVITY is the gravity vector g(q).
# CORIOLIS is the Coriolis force vector c(q, qd).
# M is the mass matrix M(q).

import numpy as np

def my_ctl(ctl, q, qd, q_des, qd_des, qdd_des, q_hist, q_deshist, gravity, coriolis, M):
    if ctl == 'P':
        u = np.zeros((2, 1))  # Implement your controller here
    elif ctl == 'PD':
        u = np.zeros((2, 1))  # Implement your controller here
    elif ctl == 'PID':
        u = np.zeros((2, 1))  # Implement your controller here
    elif ctl == 'PD_Grav':
        u = np.zeros((2, 1))  # Implement your controller here
    elif ctl == 'ModelBased':
        u = np.zeros((2, 1))  # Implement your controller here
    return u
