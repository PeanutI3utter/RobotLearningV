import numpy as np
import matplotlib.pyplot as plt
from getImitationData import *
from getProMPBasis import *

def proMP (nBasis, condition=False):

    dt = 0.002
    time = np.arange(dt,3,dt)
    nSteps = len(time)
    data = getImitationData(dt, time, multiple_demos=True)
    q = data[0]
    qd = data[1]
    qdd = data[2]

    bandwidth = 0.2
    Phi = getProMPBasis( dt, nSteps, nBasis, bandwidth )

    w = ...
    mean_w = ...
    cov_w = ...
    mean_traj = ...
    std_traj = ...

    plt.figure()
    plt.fill_between(time, mean_traj - 2*std_traj, mean_traj + 2*std_traj, alpha=0.5, edgecolor='#1B2ACC', facecolor='#089FFF')
    plt.plot(time, mean_traj, color='#1B2ACC')
    plt.plot(time, q.T)
    plt.title('ProMP with ' + str(nBasis) + ' basis functions')

    #Conditioning
    if condition:
        y_d = 3
        Sig_d = 0.0002
        t_point = np.round(2300/2)

        tmp = np.dot(cov_w, Phi[:,t_point]) / (Sig_d + np.dot(Phi[:,t_point].T,np.dot(cov_w,Phi[:,t_point])))

        cov_w_new = ...
        mean_w_new = ...
        mean_traj_new = ...
        std_traj_new = ...

        plt.figure()
        plt.fill_between(time, mean_traj - 2*std_traj, mean_traj + 2*std_traj, alpha=0.5, edgecolor='#1B2ACC', facecolor='#089FFF')
        plt.plot(time,mean_traj, color='#1B2ACC')
        plt.fill_between(time, mean_traj_new - 2*std_traj_new, mean_traj_new + 2*std_traj_new, alpha=0.5, edgecolor='#CC4F1B', facecolor='#FF9848')
        plt.plot(time, mean_traj_new, color='#CC4F1B')

        sample_traj = np.dot(Phi.T,np.random.multivariate_normal(mean_w_new,cov_w_new,10).T)
        plt.plot(time,sample_traj)
        plt.title('ProMP after contidioning with new sampled trajectories')

    plt.draw_all()
    plt.pause(0.001)
