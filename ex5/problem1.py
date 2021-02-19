import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt("./spinbotdata.txt")

X = data[:9].T
Y = data[9:].T


def phi1(x):
    q_ddot_1 = x[:, 6]
    return np.array([
        q_ddot_1,
        q_ddot_1,
        np.ones_like(q_ddot_1),
        np.ones_like(q_ddot_1)
    ]).T


def phi2(x):
    return np.array([
        (x[:, 5] * x[:, 4] * x[:, 2]) + (x[:, 2] ** 2 * x[:, 7])
    ]).reshape(-1, 1)


def phi3(x):
    return (x[:, 8] - x[:, 2] * x[:, 4] ** 2).reshape(-1, 1)


Phi1 = phi1(X)
Phi2 = phi2(X)
Phi3 = phi3(X)

theta1 = np.linalg.inv(Phi1.T @ Phi1 + 1e-8 * np.eye(4)) @ Phi1.T @ Y[:, 0]
theta2 = np.linalg.inv(Phi2.T @ Phi2 + 1e-8 * np.eye(1)) @ Phi2.T @ Y[:, 1]
theta3 = np.linalg.inv(Phi3.T @ Phi3 + 1e-8 * np.eye(1)) @ Phi3.T @ Y[:, 2]


t = np.linspace(1, 100, num=100)


def plot1():
    plt.plot(t, Y[:, 0], label="Joint 1 data")
    plt.plot(t, Phi1 @ theta1, label="Joint 1 prediction")
    print(theta1)
    print(np.sum(((Phi1 @ theta1) - Y[:, 0]) ** 2))


def plot2():
    plt.plot(t, Y[:, 1], label="Joint 2 data")
    plt.plot(t, Phi2 @ theta2, label="Joint 2 prediction")
    print(theta2)
    print(np.sum(((Phi2 @ theta2) - Y[:, 1]) ** 2))


def plot3():
    plt.plot(t, Y[:, 2], label="Joint 3 data")
    plt.plot(t, Phi3 @ theta3, label="Joint 3 prediction")
    print(theta3)
    print(np.sum(((Phi3 @ theta3) - Y[:, 2]) ** 2))


plot2()
plt.legend()
plt.show()
