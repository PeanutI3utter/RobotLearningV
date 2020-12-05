import numpy as np
import matplotlib.pyplot as plt
import abc

class LQRSim(abc.ABC):
    """
    Class that simulate the LQR system
    """
    def __init__(self) -> None:
        self.s_hist = np.array([np.random.multivariate_normal(np.zeros((2,)), np.eye(2))]).reshape(2, 1)
        self.a_hist = np.zeros((0, ))
        self.T = 50
        self.A_t = np.array([
            [1, .1],
            [0 , 1]
        ])
        self.B_t = np.array([0, .1]).reshape(-1, 1)
        self.b_t = np.array([1., 0.])
        self.Sigma_T = np.array([
            [.01, 0],
            [0, .01]
        ])
        self.K_t = np.array([5, .3]).reshape(1, -1)
        self.k_t = .3
        self.H_t = 1
        self.R_t = np.array([[
            [.01, 0],
            [0, .1]
        ]] * self.T)
        self.R_t[14] = np.array([
            [100000, 0],
            [0, .1]
        ])
        self.R_t[40] = np.array([
            [100000, 0],
            [0, .1]
        ])
        self.r_t = np.append(
            np.array([[[10.], [0.]]] * 15),
            np.array([[[20.], [0.]]] * 36),
            axis=0
        ).T[0]
        self.ran = False

    @abc.abstractclassmethod
    def step_func(self, t):
        pass

    def run(self):
        """
        runs the simulation
        """
        if self.ran:
            return
        for t in range(self.T):
            action, next_state = self.step_func(t)
            self.s_hist = np.append(self.s_hist, next_state, axis=1)
            self.a_hist = np.append(self.a_hist, action)
        self.ran = True

    def calc_reward(self):
        if not self.ran:
            raise Exception("The simulation must be run before rewards can calculated")
        self.reward = np.zeros((self.T, ))
        err = (self.s_hist[:, -1] - self.r_t[:, -1]).reshape(-1, 1)
        self.reward[-1] = -err.T@self.R_t[-1]@err
        for t in range(self.T - 2, -1, -1):
            err = (self.s_hist[:, t] - self.r_t[:, t])
            self.reward[t] = -err.T@self.R_t[t]@err-self.a_hist[t].T*self.H_t*self.a_hist[t]
        return self.reward


class LQR1(LQRSim):
    def step_func(self, t):
        current_state = self.s_hist[:, t].reshape(-1, 1)
        action = -self.K_t@current_state + self.k_t
        noise = np.random.multivariate_normal(self.b_t, self.Sigma_T).reshape(-1, 1)
        next_state = self.A_t@current_state + self.B_t*action + noise
        return action, next_state


class LQR_PCONTROLLER(LQRSim):
    def __init__(self, s_des=None) -> None:
        super().__init__()
        self.s_des = s_des if s_des is not None else self.r_t

    def step_func(self, t):
        current_state = self.s_hist[:, t].reshape(-1, 1)
        action = self.K_t@(self.s_des[:, t].reshape(-1, 1) - current_state) + self.k_t
        noise = np.random.multivariate_normal(self.b_t, self.Sigma_T).reshape(-1, 1)
        next_state = self.A_t@current_state + self.B_t*action + noise
        return action, next_state


class LQR_OPTIMAL(LQRSim):
    def __init__(self, s_des=None) -> None:
        super().__init__()
        self.s_des = s_des if s_des is not None else self.r_t
        self.V_t = np.zeros_like(self.R_t)
        self.V_t[-1] = self.R_t[-1]
        self.v_t = np.zeros_like(self.r_t)
        self.v_t[:, -1] = self.R_t[-1]@self.r_t[:, -1]
        for t in range(self.T - 2, -1, -1):
            M_t = (self.B_t/(self.H_t + \
                self.B_t.T@self.V_t[t+1]@self.B_t))@self.B_t.T@self.V_t[t + 1]@self.A_t
            self.V_t[t] = self.R_t[t] + (self.A_t - M_t).T@self.V_t[t + 1]@self.A_t
            self.v_t[:, t] = (self.R_t[t]@self.r_t[:, t].reshape(-1, 1) + \
                (self.A_t - M_t).T@(self.v_t[:, t + 1].reshape(-1, 1) - self.V_t[t + 1]@self.b_t.reshape(-1, 1))).reshape(-1)

    def step_func(self, t):
        current_state = self.s_hist[:, t].reshape(-1, 1)
        action = -(self.H_t+self.B_t.T@self.V_t[t]@self.B_t)**-1*self.B_t.T@(self.V_t[t]@(self.A_t@current_state + self.b_t.reshape(-1, 1)) - self.v_t[:, t].reshape(-1, 1))
        noise = np.random.multivariate_normal(self.b_t, self.Sigma_T).reshape(-1, 1)
        next_state = self.A_t@current_state + self.B_t*action + noise
        return action, next_state
    
class Plotter:
    i = 0

    def plot(self, sims, s1=True, s2=True, a=True, r=True, prefix=''):
        to_plot = {}
        if s1:
            to_plot['s1'] = self.i
            self.i += 1

        if s2:
            to_plot['s2'] = self.i
            self.i += 1

        if a:
            to_plot['a'] = self.i
            self.i += 1

        if r:
            to_plot['r'] = self.i
            self.i += 1

        time_states = np.linspace(0, 50, 51)
        time_r_a = np.linspace(0, 49, 50)
        [plt.figure(ii) for ii in range(self.i)]


        states = np.array([sim.s_hist for sim in sims])
        states_mean = np.mean(states, axis=0)
        states_var = np.var(states, axis=0)

        actions = np.array([sim.a_hist for sim in sims])
        action_mean = np.mean(actions, axis=0)
        action_var = np.var(actions, axis=0)

        cumulative_rewards = np.array([np.cumsum(sim.calc_reward()) for sim in sims])
        reward_mean = np.mean(cumulative_rewards, axis=0)

        if s1:
            plt.figure(to_plot['s1'])
            plt.title(f'[{prefix}] Mean of First dimension of state over time')
            plt.plot(time_states, states_mean[0])
            plt.fill_between(time_states, states_mean[0] - 2 * np.sqrt(states_var[0]), states_mean[0] + 2 * np.sqrt(states_var[0]), color='orange', alpha=.4)
            plt.xlabel('Time t')
            plt.ylabel('First dimension of the state')

        if s2:
            plt.figure(to_plot['s2'])
            plt.title(f'[{prefix}] Mean of Second dimension of state over time')
            plt.plot(time_states, states_mean[1])
            plt.fill_between(time_states, states_mean[1] - 2 * np.sqrt(states_var[1]), states_mean[1] + 2 * np.sqrt(states_var[1]), color='orange', alpha=.4)
            plt.xlabel('Time t')
            plt.ylabel('Second dimension of the state')

        if a:
            plt.figure(to_plot['a'])
            plt.title(f'[{prefix}] Mean Action over time')
            plt.plot(time_r_a, action_mean)
            plt.fill_between(time_r_a, action_mean - 2 * np.sqrt(action_var), action_mean + 2 * np.sqrt(action_var), color='orange', alpha=.4)
            plt.xlabel('Time t')
            plt.ylabel('Action')

        if r:
            plt.figure(to_plot['r'])
            plt.title(f'[{prefix}] Mean Cumulative Reward over time')
            plt.plot(time_r_a, reward_mean)
            plt.xlabel('Time t')
            plt.ylabel('Reward')


if __name__ == "__main__":
    plotter = Plotter()
    lqr = [LQR1() for _ in range(20)]
    for sim in lqr:
        sim.run()
    plotter.plot(lqr, prefix='Zero control')
    zeros = [np.zeros((2, 50)) for _ in range(20)]

    lqrp = [LQR_PCONTROLLER() for _ in range(20)]
    for sim in lqrp:
        sim.run()
    plotter.plot(lqrp, prefix='P controller')

    lqro = [LQR_OPTIMAL() for _ in range(20)]
    for sim in lqro:
        sim.run()
    plotter.plot(lqro, prefix='Optimal value')
    plt.show()
