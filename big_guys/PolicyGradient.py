import torch
import torch.nn as nn
from torch.optim import Adam
from torch.distributions.categorical import Categorical
import numpy as np
import gym
from gym.spaces import Discrete, Box


def mlp(sizes, activation=nn.Tanh, output_activation=nn.Identity):
    layers = []
    for i in range(len(sizes) - 1):
        act = activation if i < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[i], sizes[i + 1]), act()]
    return nn.Sequential(*layers)


def reward_to_go(rews):
    n = len(rews)
    rtgs = np.zeros_like(rews)
    for i in reversed(range(n)):
        rtgs[i] = rews[i] + (rtgs[i + 1] if i + 1 < n else 0)
    return list(rtgs)


def train(env_name="CartPole-v0", hidden_sizes=[32], lr=1e-2,
          epochs=50, batch_size=5000, render=False, rtg=False):
    env = gym.make(env_name)
    assert isinstance(env.observation_space, Box)
    assert isinstance(env.action_space, Discrete)

    obs_dim = env.observation_space.shape[0]
    n_acts = env.action_space.n

    logits_net = mlp(sizes=[obs_dim] + hidden_sizes + [n_acts])

    # returns probability distribution
    def get_policy(obs):
        logits = logits_net(obs)
        return Categorical(logits=logits)

    def get_action(obs):
        return get_policy(obs).sample().item()

    def compute_loss(obs, act, weights):
        logp = get_policy(obs).log_prob(act)
        return -(logp * weights).mean()

    optimizer = Adam(logits_net.parameters(), lr=lr)

    def train_one_epoch():
        batch_obs = []
        batch_acts = []
        batch_weights = []
        batch_rets = []
        batch_lens = []

        # first observation comes from starting distribution
        obs, done, ep_rews = env.reset(), False, []

        # render first episode of each epoch
        finished_rendering_this_epoch = False

        while True:
            if (not finished_rendering_this_epoch) and render:
                env.render()

            batch_obs.append(obs.copy())

            act = get_action(torch.as_tensor(obs, dtype=torch.float32))
            obs, rew, done, _ = env.step(act)

            batch_acts.append(act)
            ep_rews.append(rew)

            if done:
                ep_ret, ep_len = sum(ep_rews), len(ep_rews)
                batch_rets.append(ep_ret)
                batch_lens.append(ep_len)

                # each action has weight that equal to sum of all rewards obtained in episode
                if rtg:
                    batch_weights += reward_to_go(ep_rews)
                else:
                    batch_weights += [ep_ret] * ep_len

                obs, done, ep_rews = env.reset(), False, []
                finished_rendering_this_epoch = True

                if len(batch_obs) > batch_size:
                    break

        optimizer.zero_grad()
        batch_loss = compute_loss(obs=torch.as_tensor(batch_obs, dtype=torch.float32),
                                  act=torch.as_tensor(batch_acts, dtype=torch.int32),
                                  weights=torch.as_tensor(batch_weights, dtype=torch.float32))
        batch_loss.backward()
        optimizer.step()
        return batch_loss, batch_rets, batch_lens

    loss, rets, lens = [], [], []
    for i in range(epochs):
        batch_loss, batch_rets, batch_lens = train_one_epoch()
        loss.append(batch_loss)
        rets.append(np.mean(batch_rets))
        lens.append(np.mean(batch_lens))
        # print("epoch: %3d \t loss: %.3f \t return: %.3f \t ep_len: %.3f" %
        #       (i, batch_loss, np.mean(batch_rets), np.mean(batch_lens)))

    return loss, rets, lens


import matplotlib.pyplot as plt

_, vanila_rets, vanila_lens = train(epochs=30)
_, rtg_rets, rtg_lens = train(epochs=30, rtg=True)

plt.plot(vanila_rets, label="Vanila")
plt.plot(rtg_rets, label="rtg")
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Episode Return")
plt.show()

plt.plot(vanila_lens, label="Vanila")
plt.plot(rtg_lens, label="rtg")
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Episode Length")
plt.show()
