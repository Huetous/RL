import numpy as np
from collections import defaultdict


def sample_policy(obs):
    score, dealer_score, usable_ace = obs
    return 0 if score >= 20 else 1


def get_one_episode(env, pi, max_ep_len):
    episode = []
    s = env.reset()
    for t in range(max_ep_len):
        a = pi(s)
        s2, r, done, _ = env.step(a)
        episode.append((s, a, r))
        if done:
            break
        s = s2
    return episode


def compute_return(gamma, timesteps):
    return sum([x[2] * (gamma ** i) for i, x in enumerate(timesteps)])


def make_epsilon_greedy_policy(Q, epsilon, nA):
    def policy_fn(obs):
        A = np.ones(nA, dtype=float) * epsilon / nA
        best_action = np.argmax(Q[obs])
        A[best_action] += (1.0 - epsilon)
        return np.random.choice(np.arange(len(A)), p=A)

    return policy_fn


def MC_prediction(env, pi, gamma=1.0, max_ep_len=100, num_episodes=2000, first_visit=True):
    V = defaultdict(float)
    N = defaultdict(float)

    for epoch in range(num_episodes):

        episode = get_one_episode(env, pi, max_ep_len)

        visited_states = set()
        for t in range(len(episode)):
            s, a, r = episode[t]
            s = tuple(s)

            if first_visit and s in visited_states:
                continue

            G = compute_return(gamma, episode[t:])
            N[s] += 1
            V[s] += (1 / N[s]) * (G - V[s])

            visited_states.add(s)

    return V


def MC_on_policy_control(env, gamma=1.0, epsilon=0.1, max_ep_len=100, num_episodes=2000, first_visit=True):
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    N = defaultdict(lambda: np.zeros(env.action_space.n))
    pi = make_epsilon_greedy_policy(Q, epsilon, env.action_space.n)

    for i in range(num_episodes):

        episode = get_one_episode(env, pi, max_ep_len)

        visited_pairs = set()
        for t in range(len(episode)):
            s, a, r = episode[t]
            pair = (s, a)

            if first_visit and pair in visited_pairs:
                continue

            G = compute_return(gamma, episode[t:])
            N[s][a] += 1
            Q[s][a] += (1 / N[s][a]) * (G - Q[s][a])

            visited_pairs.add(pair)
    return Q, pi


def MC_off_policy_control_imp_sampling(env, gamma=1.0, epsilon=0.1, max_ep_len=100, num_episodes=2000,
                                       first_visit=True):
    def get_behaviour_policy_probs(Q, epsilon, nA):
        def policy_fn(obs):
            A = np.ones(nA, dtype=float) * epsilon / nA
            best_action = np.argmax(Q[obs])
            A[best_action] += (1.0 - epsilon)
            return A

        return policy_fn

    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    C = defaultdict(lambda: np.zeros(env.action_space.n))

    mu_probs = get_behaviour_policy_probs(Q, epsilon, env.action_space.n)
    for ep in range(num_episodes):

        episode = []
        s = env.reset()
        for t in range(max_ep_len):
            probs = mu_probs(s)
            a = np.random.choice(np.arange(len(probs)), p=probs)
            s2, r, done, _ = env.step(a)
            episode.append((s, a, r))
            if done:
                break
            s = s2

        W = 1
        for t in range(len(episode)):
            s, a, r = episode[t]

            G = compute_return(gamma, episode[t:])
            C[s][a] += W
            Q[s][a] += (W / C[s][a]) * (G - Q[s][a])

            if np.argmax(Q[s][a]) is not a:
                break
            W /= mu_probs(s)[a]

    return Q
# V_10k = MC_prediction(env=BlackjackEnv(), pi=sample_policy, num_episodes=100000)

# Q, pi = MC_on_policy_control(env=BlackjackEnv(), num_episodes=500000, epsilon=0.1)
# V = defaultdict(float)
# for state, actions in Q.items():
#     action_value = np.max(actions)
#     V[state] = action_value
# plotting.plot_value_function(V, title="Optimal Value Function")

# Q = MC_off_policy_control_imp_sampling(env=BlackjackEnv(), num_episodes=100000)
# V = defaultdict(float)
# for state, actions in Q.items():
#     action_value = np.max(actions)
#     V[state] = action_value
# plotting.plot_value_function(V, title="Optimal Value Function")
