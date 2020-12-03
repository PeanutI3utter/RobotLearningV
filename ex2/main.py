import numpy as np
import matplotlib as plt

class LQRSim:
    """
    Class that simulate the LQR system
    """
    def __init__(self) -> None:
        self.s_hist = np.array([np.random.multivariate_normal(np.zeros((2,)), np.eye(2))]).reshape(2, 1)
        self.a_hist = np.zeros((1, 0))
        self.T = 50
        self.A_t = np.array([
            [1, .1],
            [0 , 1]
        ])
        self.B_t = np.array([0, .1]).reshape(-1, 1)
        self.b_t = np.array([5., 0.])
        self.Sigma_T = np.array([
            [.01, 0],
            [0, .01]
        ])
        self.K_t = np.array([5, .3]).reshape(1, -1)
        self.k_t = .3
        self.H_t = 1
        self.R_t = np.array([[
            [.01, 0],
            [0, .01]
        ]] * self.T)
        self.R_t[14] = np.array([
            [100000, 0],
            [0, .01]
        ])
        self.R_t[40] = np.array([
            [100000, 0],
            [0, .01]
        ])
        self.r_t = np.append(
            np.array([[[10.], [0.]]] * 15),
            np.array([[[20.], [0.]]] * 36),
            axis=0
        )

    def step_func(self, t):
        return None, None

    def run(self):
        """
        runs the simulation
        """
        for t in range(self.T):
            action, next_state = self.step_func(t)
            self.s_hist = np.append(self.s_hist, next_state, axis=1)
            self.a_hist = np.append(self.a_hist, action, axis=1)


class LQR1(LQRSim):
    def step_func(self, t):
        current_state = self.s_hist[:, t].reshape(-1, 1)
        action = -self.K_t@current_state + self.k_t
        noise = np.random.multivariate_normal(self.b_t, self.Sigma_T).reshape(-1, 1)
        next_state = self.A_t@current_state + self.B_t*action + noise
        return action, next_state
    

        
if __name__ == "__main__":
    test = LQR1()
    test.run()
    print(test.s_hist)
    