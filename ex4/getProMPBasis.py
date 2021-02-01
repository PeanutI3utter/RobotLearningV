import numpy as np
import matplotlib.pyplot as plt


def getProMPBasis(dt, nSteps, n_of_basis, bandwidth):

    time = np.arange(dt, nSteps * dt, dt)
    basis = np.linspace(-2 * bandwidth, dt * nSteps + 2 * bandwidth, n_of_basis)
    Phi = np.zeros((nSteps, n_of_basis))
    for i, t in enumerate(time):
        for j in range(n_of_basis):
            Phi[i, j] = np.exp(-0.5 * (t - basis[j]) ** 2 / bandwidth ** 2)
        Phi[i] = Phi[i] / np.sum(Phi[i])

    return Phi
