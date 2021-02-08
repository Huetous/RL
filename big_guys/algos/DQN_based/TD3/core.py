import torch
import torch.nn as nn
import toss.RL.algos.DDPG.core as core

class MLPActorCritic(nn.Module):
    def __init__(self, observation_space, action_space, hidden_sizes=(256, 256),
                 activation=nn.ReLU):
        super().__init__()

        obs_dim = observation_space.shape[0]
        act_dim = action_space.shape[0]
        act_limit = action_space.high[0]

        self.pi = core.MLPActor(obs_dim, act_dim, hidden_sizes, activation, act_limit)
        self.q1 = core.MLPQFunction(obs_dim, act_dim, hidden_sizes, activation)
        self.q2 = core.MLPQFunction(obs_dim, act_dim, hidden_sizes, activation)

    def act(self, obs):
        with torch.no_grad():
            return self.pi(obs).numpy()

