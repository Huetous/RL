import gym
import numpy as np
from collections import defaultdict
from RL.grandpas.lib import plotting
import matplotlib.pyplot as plt


def make_epsilon_greedy_policy(Q, epsilon, nA):
    def policy_fn(obs):
        A = np.ones(nA, dtype=float) * epsilon / nA
        v = Q[obs]
        best_action = np.argmax(v)
        A[best_action] += (1.0 - epsilon)
        return np.random.choice(np.arange(len(A)), p=A)

    return policy_fn


def SARSA(env, max_ep_len=1000, num_episodes=10000, alpha=0.5, epsilon=0.1, gamma=1.0, render=False):
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    pi = make_epsilon_greedy_policy(Q, epsilon, env.action_space.n)

    stats = plotting.EpisodeStats(
        episode_lengths=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes))

    for ep in range(num_episodes):

        s = env.reset()
        a = pi(s)
        for t in range(max_ep_len):
            s2, r, done, _ = env.step(a)
            a2 = pi(s2)
            Q[s][a] += alpha * (r + gamma * Q[s2][a2] - Q[s][a])

            stats.episode_rewards[ep] += r
            stats.episode_lengths[ep] = t

            if done:
                break
            s = s2
            a = a2

        if render:
            env.render()
    return Q, stats


def Q_learning(env, max_ep_len=1000, num_episodes=10000, alpha=0.5, epsilon=0.1, gamma=1.0):
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    mu = make_epsilon_greedy_policy(Q, epsilon, env.action_space.n)

    stats = plotting.EpisodeStats(
        episode_lengths=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes))

    for ep in range(num_episodes):

        s = env.reset()
        for t in range(max_ep_len):
            a = mu(s)
            s2, r, done, _ = env.step(a)
            a2 = np.argmax(Q[s2])
            Q[s][a] += alpha * (r + gamma * Q[s2][a2] - Q[s][a])

            stats.episode_rewards[ep] += r
            stats.episode_lengths[ep] = t

            if done:
                break
            s = s2

    return Q, stats
