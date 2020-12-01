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
    K_P = np.array([60, 30]) * 10
    K_D = np.array([10, 6]) * 10
    K_I = np.array([.1, .1]) * 10
    if ctl == 'P':
        u = K_P * (q_des - q)  # Implement your controller here
    elif ctl == 'PD':
        u =  K_P * (q_des - q) + K_D * (qd_des - qd) # Implement your controller here
    elif ctl == 'PID':
        I =  K_I * np.sum(q_deshist - q_hist, axis=0).reshape(-1) if q_deshist.shape[0] > 0 else np.array([0, 0])
        u = K_P * (q_des - q) + K_D * (qd_des - qd) + I  # Implement your controller here
    elif ctl == 'PD_Grav':
        u =  K_P * (q_des - q) + K_D * (qd_des - qd) + gravity # Implement your controller here
    elif ctl == 'ModelBased':
        q_ref = qdd_des + K_P * (q_des - q) + K_D * (qd_des - qd)
        u = M @ q_ref + coriolis + gravity  # Implement your controller here
    return u.reshape(-1, 1)
