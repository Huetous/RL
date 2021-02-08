import numpy as np
from grandpas.lib.envs.gridworld import GridworldEnv
import matplotlib.pyplot as plt




class DP:
    def __init__(self, env, gamma=1.0):
        self.env = env
        self.obs_dim = env.nS
        self.act_dim = env.nA
        self.P = env.P
        self.gamma = gamma

    def Q(self, s, a, V):
        prob, s2, r, _ = self.P[s][a][0]
        return r + self.gamma * prob * V[s2]

    def policy_evaluation(self, pi, theta=1e-5, max_epochs=2000):
        V = np.zeros(self.obs_dim)

        for epoch in range(max_epochs):
            delta = 0

            for s in range(self.obs_dim):
                v = 0
                for a in range(self.act_dim):
                    v += pi[s][a] * self.Q(s, a, V)

                delta = max(delta, abs(v - V[s]))
                V[s] = v

            if delta < theta:
                break

        return np.array(V)

    def policy_iteration(self, max_epochs=2000):
        pi = np.ones([self.obs_dim, self.act_dim]) / self.act_dim

        for epoch in range(max_epochs):
            V = self.policy_evaluation(pi)

            pi_stable = True
            for s in range(self.obs_dim):
                a = np.argmax(pi[s])

                Qs = [self.Q(s, a, V) for a in range(self.act_dim)]
                best_a = np.argmax(Qs)

                if a != best_a:
                    pi_stable = False
                pi[s] = np.zeros(self.act_dim)
                pi[s][best_a] = 1

            if pi_stable:
                return pi, V

        return pi, np.zeros(self.obs_dim)

    def value_iteration(self, theta=1e-5, max_epochs=2000):
        V = np.zeros(self.obs_dim)
        pi = np.zeros([self.obs_dim, self.act_dim])

        for epoch in range(max_epochs):
            delta = 0

            for s in range(self.obs_dim):
                Qs = [self.Q(s, a, V) for a in range(self.act_dim)]
                opt_act = np.argmax(Qs)
                v = self.Q(s, opt_act, V)

                delta = max(delta, abs(v - V[s]))

                V[s] = v
                pi[s] = np.zeros(self.act_dim)
                pi[s][opt_act] = 1

            if delta < theta:
                break

        return pi, V


def ex1():
    env = GridworldEnv()
    dp = DP(env)
    random_policy = np.ones([env.nS, env.nA]) / env.nA
    v = dp.policy_evaluation(random_policy)

    expected_v = np.array([0, -14, -20, -22, -14, -18,
                           -20, -20, -20, -20, -18,
                           -14, -22, -20, -14, 0])
    np.testing.assert_array_almost_equal(v, expected_v, decimal=2)

    policy, v = dp.policy_iteration()
    print(np.reshape(np.argmax(policy, axis=1), env.shape))
    print(np.reshape(v, env.shape))

    opt_policy, opt_value_fn = dp.value_iteration()
    print(np.reshape(np.argmax(opt_policy, axis=1), env.shape))
    print(np.reshape(opt_value_fn, env.shape))

ex1()

def ex2():
    def value_iteration_for_gamblers(p_h, gamma=1.0, theta=1e-5, max_iters=2000):
        def one_step_lookahead(s, V, rewards):
            A = np.zeros(101)
            stakes = range(1, min(s, 100 - s) + 1)

            for a in stakes:
                A[a] = p_h * (rewards[s + a] + gamma * V[s + a]) + (1 - p_h) * (rewards[s - a] + gamma * V[s - a])
            return A

        nS = 101
        rewards = np.zeros(nS)
        rewards[100] = 1

        V = np.zeros(nS)
        policy = np.zeros(nS)

        iters_count = 0
        while iters_count < max_iters:
            delta = 0

            for s in range(nS):
                A = one_step_lookahead(s, V, rewards)
                v = max(A)
                optimal_action = np.argmax(A)

                delta = max(delta, abs(v - V[s]))

                V[s] = v
                policy[s] = optimal_action

            if delta < theta:
                break
            iters_count += 1

        return policy, V

    policy, V = value_iteration_for_gamblers(0.25)
    print(policy)
    print(V)

    x = range(100)
    y = V[:100]
    plt.plot(x, y)
    plt.xlabel('Capital')
    plt.ylabel('Value Estimates')
    plt.title('Final Policy (action stake) vs State (Capital)')
    plt.show()

    y = policy[1:]
    plt.bar(x, y, align='center', alpha=0.5)
    plt.xlabel('Capital')
    plt.ylabel('Final policy (stake)')
    plt.title('Capital vs Final Policy')
    plt.show()
