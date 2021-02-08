from tqdm import tqdm
from RL.big_guys.algos.DQN_based.DQN.ReplayBuffer import UniformReplayBuffer, Stats, RankBasedPrioritizedReplay
from RL.big_guys.algos.DQN_based.DQN.Agent import Agent

import numpy as np
import gym
import torch


def DQN(env_name, seed=0, epochs=100, max_ep_len=1000, gamma=0.99, q_lr=1e-3,
        steps_per_epoch=3000, start_steps=5000, update_after=100, update_every=4,
        buffer_size=10000, batch_size=32, eps_sched=dict(), update_target_every=100,
        k_frames=4, double=False, dueling=False, render=False,
        priority=None, buf_kwargs=dict()):
    assert priority in [None, "rank", "prop"]

    torch.manual_seed(seed)
    np.random.seed(seed)

    env = gym.make(env_name)
    assert isinstance(env.action_space, gym.spaces.Discrete), "Action space has to be Discrete"
    obs_dim = env.observation_space.shape
    n_actions = env.action_space.n

    buf_cls = UniformReplayBuffer
    if priority is "rank":
        buf_cls = RankBasedPrioritizedReplay
    if priority is "prop":
        pass

    buf = buf_cls(buffer_size, obs_dim, k_frames, **buf_kwargs)
    agent = Agent(obs_dim[-1] * k_frames, n_actions, q_lr, gamma, eps_sched,
                  double=double, dueling=dueling)

    stats = Stats(EpRet=[], EpLen=[])
    total_steps = steps_per_epoch * epochs

    o, ep_ret, ep_len = env.reset(), 0, 0
    for t in tqdm(range(total_steps)):
        if render:
            env.render()
        buf.store_obs(o)  # store image
        o = buf.get_recent_context()  # take last k_frames images

        eps = None if t > start_steps else 1.  # eps = 1 means that we always perform sample of actions
        a = agent.act(o, eps)

        o2, r, d, _ = env.step(a)

        ep_ret += r
        ep_len += 1

        d = False if ep_len == max_ep_len else d
        r = np.clip(r, -1., 1.)  # reward clipping, noted in original paper
        buf.store_effect(a, r, d)  # store a, r, d for previously stored (on current iter) image

        o = o2

        if d or (ep_len == max_ep_len):
            stats.EpRet.append(ep_ret)
            stats.EpLen.append(ep_len)
            o, ep_ret, ep_len = env.reset(), 0, 0

        if t >= update_after and t % update_every == 0:
            batch = buf.sample_batch(batch_size)
            changes = agent.update_main(batch)
            if changes is not None and isinstance(buf, RankBasedPrioritizedReplay):
                buf.update_priorities(changes)

        if (t + 1) % update_target_every == 0:
            agent.update_target()

    return stats, agent


DQN(env_name="Berzerk-v0", epochs=1, priority="rank", buf_kwargs={"alpha": 0.5},
    eps_sched={"start": 0.9, "end": 0.001, "decay": 1000})
