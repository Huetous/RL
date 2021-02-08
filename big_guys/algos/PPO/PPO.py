import RL.big_guys.algos.core as core
import torch
import numpy as np
import gym
from torch.optim import Adam
from RL.big_guys.utils.log import EpochLogger
import time
import matplotlib.pyplot as plt

def PPO_clip(env_name, actor_critic=core.MLPActorCritic, ac_kwargs=dict(), seed=0,
             steps_per_epoch=3000, epochs=30, gamma=0.99, clip_ratio=0.2, pi_lr=3e-4,
             vf_lr=1e-3, train_pi_iters=80, train_v_iters=80, lam=0.97, max_ep_len=1000,
             target_kl=0.01, logger_kwargs=dict(), save_freq=10):
    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    torch.manual_seed(seed)
    np.random.seed(seed)

    env = gym.make(env_name)
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape

    ac = actor_critic(env.observation_space, env.action_space, **ac_kwargs)

    var_counts = tuple(core.count_vars(module) for module in [ac.pi, ac.v])
    logger.log('\nNumber of parameters: \t pi: %d, \t v: %d\n' % var_counts)

    buf = core.Buffer(obs_dim, act_dim, steps_per_epoch, gamma, lam)

    def compute_loss_pi(data):
        obs, act, adv, logp_old = data["obs"], data["act"], data["adv"], data["logp"]

        pi, logp = ac.pi(obs, act)
        ratio = torch.exp(logp - logp_old)
        clip_adv = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * adv
        loss_pi = -(torch.min(ratio * adv, clip_adv)).mean()

        approx_kl = (logp_old - logp).mean().item()
        ent = pi.entropy().mean().item()
        clipped = ratio.gt(1 + clip_ratio) | ratio.lt(1 - clip_ratio)  # mask
        clipfrac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()
        pi_info = dict(kl=approx_kl, ent=ent, cf=clipfrac)

        return loss_pi, pi_info

    def compute_loss_v(data):
        obs, ret = data["obs"], data["ret"]
        return ((ac.v(obs) - ret) ** 2).mean()

    pi_opt = Adam(ac.pi.parameters(), lr=pi_lr)
    vf_opt = Adam(ac.v.parameters(), lr=vf_lr)

    def update():
        data = buf.get()

        pi_loss_old, pi_info_old = compute_loss_pi(data)
        pi_loss_old = pi_loss_old.item()
        v_loss_old = compute_loss_v(data).item()

        for i in range(train_pi_iters):
            pi_opt.zero_grad()
            loss_pi, pi_info = compute_loss_pi(data)
            kl = pi_info["kl"]
            if kl > 1.5 * target_kl:
                break  # early stopping
            loss_pi.backward()
            pi_opt.step()

        logger.store(StopIter=i)

        for i in range(train_v_iters):
            vf_opt.zero_grad()
            loss_v = compute_loss_v(data)
            loss_v.backward()
            vf_opt.step()

        kl, ent, cf = pi_info["kl"], pi_info_old["ent"], pi_info["cf"]
        logger.store(LossPi=pi_loss_old, LossV=v_loss_old,
                     KL=kl, Entropy=ent, ClipFrac=cf,
                     DeltaLossPi=(loss_pi.item() - pi_loss_old),
                     DeltaLossV=(loss_v.item() - v_loss_old))

    start_time = time.time()
    o, ep_ret, ep_len = env.reset(), 0, 0
    for epoch in range(epochs):
        for t in range(steps_per_epoch):
            a, v, logp = ac.step(torch.as_tensor(o, dtype=torch.float32))

            next_o, r, d, _ = env.step(a)
            ep_ret += r
            ep_len += 1

            buf.store(o, a, r, v, logp)
            logger.store(VVals=v)

            o = next_o

            timeout = ep_len == max_ep_len
            terminal = d or timeout
            epoch_ended = t == steps_per_epoch - 1

            if terminal or epoch_ended:
                if epoch_ended and not terminal:
                    print('Warning: trajectory cut off by epoch at %d steps.' % ep_len, flush=True)
                if timeout or epoch_ended:
                    _, v, _ = ac.step(torch.as_tensor(o, dtype=torch.float32))
                else:
                    v = 0
                buf.finish_path(v)
                if terminal:
                    logger.store(EpRet=ep_ret, EpLen=ep_len)
                o, ep_ret, ep_len = env.reset(), 0, 0
        update()

        logger.log_tabular('Epoch', epoch)
        logger.log_tabular('EpRet', with_min_and_max=True)
        logger.log_tabular('EpLen', average_only=True)
        logger.log_tabular('VVals', with_min_and_max=True)
        logger.log_tabular('TotalEnvInteracts', (epoch + 1) * steps_per_epoch)
        logger.log_tabular('LossPi', average_only=True)
        logger.log_tabular('LossV', average_only=True)
        logger.log_tabular('DeltaLossPi', average_only=True)
        logger.log_tabular('DeltaLossV', average_only=True)
        logger.log_tabular('Entropy', average_only=True)
        logger.log_tabular('KL', average_only=True)
        logger.log_tabular('ClipFrac', average_only=True)
        logger.log_tabular('StopIter', average_only=True)
        logger.log_tabular('Time', time.time() - start_time)
        logger.dump_tabular()

    logger.save_to_csv()

PPO_clip(env_name='CartPole-v0')




