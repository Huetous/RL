import time
from torch.optim import Adam
import itertools
from RL.big_guys.algos.DDPG.DDPG import ReplayBuffer
from RL.big_guys.algos.TD3 import core
from copy import deepcopy
from RL.big_guys.utils.log import EpochLogger
import torch
import numpy as np


def TD3(env_fn, actor_critic=core.MLPActorCritic, ac_kwargs=dict(), seed=0,
        steps_per_epoch=4000, epochs=100, replay_size=int(1e6), gamma=0.99,
        polyak=0.995, pi_lr=1e-3, q_lr=1e-3, batch_size=100, start_steps=10000,
        update_after=1000, update_every=50, act_noise=0.1, target_noise=0.2,
        noise_clip=0.5, policy_delay=2, num_test_episodes=10, max_ep_len=1000,
        logger_kwargs=dict(), save_freq=1):
    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    torch.manual_seed(seed)
    np.random.seed(seed)

    env, test_env = env_fn(), env_fn()
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape[0]

    act_limit = env.action_space.high[0]

    ac = actor_critic(env.observation_space, env.action_space, **ac_kwargs)
    ac_targ = deepcopy(ac)

    for p in ac_targ.parameters():
        p.requires_grad = False

    q_params = itertools.chain(ac.q1.parameters(), ac.q2.parameters())

    replay_buf = ReplayBuffer(obs_dim, act_dim, replay_size)

    def compute_loss_q(data):
        o, a, r, o2, d = data["obs"], data["act"], data["rew"], data["obs2"], data["done"]

        q1 = ac.q1(o, a)
        q2 = ac.q2(o, a)

        with torch.no_grad():
            pi_targ = ac_targ.pi(o2)

            eps = torch.rand_like(pi_targ) * target_noise
            eps = torch.clamp(eps, -noise_clip, noise_clip)
            a2 = pi_targ + eps
            a2 = torch.clamp(a2, -act_limit, act_limit)

            q1_pi_targ = ac_targ.q1(o2, a2)
            q2_pi_targ = ac_targ.q2(o2, a2)
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
            backup = r + gamma * (1 - d) * q_pi_targ

        loss_q1 = ((q1 - backup) ** 2).mean()
        loss_q2 = ((q2 - backup) ** 2).mean()
        loss_q = loss_q1 + loss_q2

        loss_info = dict(Q1Vals=q1.detach().numpy(), Q2Vals=q2.detach().numpy())
        return loss_q, loss_info

    def compute_loss_pi(data):
        o = data["obs"]
        q1_pi = ac.q1(o, ac.pi(o))
        return -q1_pi.mean()

    pi_opt = Adam(ac.pi.parameters(), lr=pi_lr)
    q_opt = Adam(q_params, lr=q_lr)

    def update(data, timer):
        q_opt.zero_grad()
        loss_q, loss_info = compute_loss_q(data)
        loss_q.backward()
        q_opt.step()

        logger.store(LossQ=loss_q.item(), **loss_info)

        if timer % policy_delay == 0:

            # Freeze Q-networks so you don't waste computational effort
            # computing gradients for them during the policy learning step.
            for p in q_params:
                p.requires_grad = False

            pi_opt.zero_grad()
            loss_pi = compute_loss_pi(data)
            loss_pi.backward()
            pi_opt.step()

            for p in q_params:
                p.requires_grad = True

            logger.store(LossPi=loss_pi.item())

            with torch.no_grad():
                for p, p_targ in zip(ac.parameters(), ac_targ.parameters()):
                    p_targ.data.mul_(polyak)
                    p_targ.data.add_((1 - polyak) * p.data)

    def get_action(o, noise_scale):
        a = ac.act(torch.as_tensor(o, dtype=torch.float32))
        a += noise_scale * np.random.randn(act_dim)
        return np.clip(a, -act_limit, act_limit)

    def test_agent():
        for j in range(num_test_episodes):
            o, d, ep_ret, ep_len = test_env.reset(), False, 0, 0
            while not (d or (ep_len == max_ep_len)):
                o, r, d, _ = test_env.step(get_action(o, 0))
                ep_ret += r
                ep_len += 1
            logger.store(TestEpRet=ep_ret, TestEpLen=ep_len)

    total_steps = steps_per_epoch * epochs
    start_time = time.time()
    o, ep_ret, ep_len = env.reset(), 0, 0

    for t in range(total_steps):
        if t > start_steps:
            a = get_action(o, act_noise)
        else:
            a = env.action_space.sample()

        o2, r, d, _ = env.step(a)
        ep_ret += r
        ep_len += 1

        d = False if ep_len == max_ep_len else d

        replay_buf.store(o, a, r, o2, d)

        o = o2

        if d or (ep_len == max_ep_len):
            logger.store(EpRet=ep_ret, EpLen=ep_len)
            o, ep_ret, ep_len = env.reset(), 0, 0

        if t >= update_after and t % update_every == 0:
            for j in range(update_every):
                batch = replay_buf.sample_batch(batch_size)
                update(data=batch, timer=j)

        if (t + 1) % steps_per_epoch == 0:
            epoch = (t + 1) // steps_per_epoch

            if (epoch % save_freq == 0) or (epoch == epochs):
                logger.save_state({'env': env}, None)

            test_agent()

            logger.log_tabular('Epoch', epoch)
            logger.log_tabular('EpRet', with_min_and_max=True)
            logger.log_tabular('TestEpRet', with_min_and_max=True)
            logger.log_tabular('EpLen', average_only=True)
            logger.log_tabular('TestEpLen', average_only=True)
            logger.log_tabular('TotalEnvInteracts', t)
            logger.log_tabular('Q1Vals', with_min_and_max=True)
            logger.log_tabular('Q2Vals', with_min_and_max=True)
            logger.log_tabular('LossPi', average_only=True)
            logger.log_tabular('LossQ', average_only=True)
            logger.log_tabular('Time', time.time() - start_time)
            logger.dump_tabular()
