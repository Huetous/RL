import numpy as np


class DebiasedConstantStep:
    def __init__(self, alpha):
        self.base = 0
        self.alpha = alpha

    def __call__(self, action):
        self.base = self.alpha + (1 - self.alpha) * self.base
        return self.alpha / self.base


class SoftmaxDistribution:
    def __init__(self, n_actions, alpha, baseline):
        self.H = np.zeros(n_actions)
        self.avg_reward = 0
        self.alpha = alpha
        self.baseline = baseline

    def update(self, action, reward, t):
        prob = np.exp(self.H[action]) / sum(self._powers())
        mask = np.array([a != action for a in range(len(self.H))])

        self.H -= mask * self.alpha * (reward - self.avg_reward) * prob
        self.H[action] += self.alpha * (reward - self.avg_reward) * (1 - prob)

        if self.baseline:
            # We use timestamp as a counter of how many times actions were selected. We obtain average reward
            # with the sample average method. We use t + 1, to avoid division by zero.
            self.avg_reward += (1 / (t + 1)) * (reward - self.avg_reward)

    def sample_action(self):
        return np.random.choice(range(len(self.H)), p=self._powers() / sum(self._powers()))

    def _powers(self):
        return list(map(np.exp, self.H))


class Agent:
    """
    Implements estimator of expected value via empirical mean estimation
    """

    def __init__(self, eps=0.1, alpha=None, n_actions=10, init_value=0,
                 debiased=False, UCB=False, c=2, preference=False, baseline=False):
        if preference:  # instead of estimating the action values we can directly favor particular actions
            self.distr = SoftmaxDistribution(n_actions, alpha, baseline)
        else:
            # action-value estimator for each action
            self.Q = np.ones(n_actions) * init_value

        self.n_actions = n_actions
        self.eps = eps
        self.alpha = alpha  # step-size

        self.preference = preference
        self.UCB = UCB  # Upper Confidence Bound
        self.c = c  # the degree of exploration

        if alpha is None or UCB:
            self.N = np.zeros(n_actions)  # number of times action has been selected

        if UCB: assert c > 0, "The degree of exploration 'c' must be greater than 0"

        if debiased:
            self.step = DebiasedConstantStep(alpha)  # Get rid of initial bias when using constant step-size
        else:
            self.step = lambda a: (1 / self.N[a] if alpha is None else alpha)

    def act(self, t):
        """
        Select action with epsilon greedy policy
        t: timestamp
        :return: action
        """
        if self.preference:
            return self.distr.sample_action()

        if np.random.random() > self.eps:
            ucb = 0  # additional term is used only when UCB is used
            if self.UCB and t > 0:
                # self.N + 0.1, to avoid division by zero and also to encourage selection of action
                # that haven't been selected before
                ucb = self.c * np.sqrt(np.log(t) / (self.N + 0.1))
            return np.argmax(self.Q + ucb)
        else:
            return np.random.choice(self.n_actions)

    def update(self, action, reward, t):
        """
        Update action value estimates
        """
        if self.preference:
            self.distr.update(action, reward, t)

        else:
            if self.alpha is None or self.UCB:
                self.N[action] += 1  # increase counter of visited actions

            # update estimate for visited actions
            self.Q[action] += self.step(action) * (reward - self.Q[action])


class Experiment:
    """
    Multi-armed bandit problem: k actions, 1 state, goal is to maximize expected reward.
    True values of actions are selected from normal distribution with 0 mean and variance 1.
    Then those values are used as means for reward distributions of corresponding actions.
    """

    def __init__(self, agents_kwargs, n_actions=10, stationary=True, true_values_init=0):
        # create action-value estimators for each value of epsilon
        self.agents = [Agent(n_actions=n_actions, **kwargs) for kwargs in agents_kwargs]
        self.preference = any([agent.preference for agent in self.agents])
        self.stationary = stationary

        # true values of actions (means of reward distributions)
        means = np.ones(n_actions) * true_values_init
        if self.stationary:
            self.action_values = np.random.normal(
                means,
                np.ones(n_actions),
                n_actions)
        else:
            self.action_values = means

        self.vars = np.ones(len(self.agents))  # (variances of reward distributions)

        self.optimal_action = np.argmax(self.action_values)

    def run(self, iters):
        """
        Runs experiment for iters iterations
        :param iters: number of iteration
        :return: rewards for each epsilon
        """

        # count how many time optimal action has been selected
        optimal_action_count = np.zeros((iters, len(self.agents)))
        rewards = np.zeros((iters, len(self.agents)))
        max_abs_approx_errors = np.zeros((iters, len(self.agents)))

        for i in range(iters):
            # select one action for each given epsilon
            actions = [agent.act(i) for agent in self.agents]

            optimal_action_count[i] = [action == self.optimal_action for action in actions]

            # take action value (mean) for each selected action
            means = [self.action_values[action] for action in actions]

            # sample rewards from len(self.epsilon) normal distributions with corresponding means and variances of 1
            rewards[i] = np.random.normal(means, self.vars, len(self.agents))

            # update action-value estimates
            for j, (a, r) in enumerate(zip(actions, rewards[i])):
                self.agents[j].update(a, r, i)
                if not self.preference:
                    max_abs_approx_errors[i][j] = max(abs(self.agents[j].Q - self.action_values))

            # if problem is not stationary then add random noise to each true action value
            # and update optimal action
            if not self.stationary:
                self.action_values += np.random.normal(0, 0.01, len(self.action_values))
                self.optimal_action = np.argmax(self.action_values)

        return rewards, optimal_action_count, max_abs_approx_errors



