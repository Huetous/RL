import big_guys.algos.core as core
import numpy as np
import torch
from torch.optim import Adam
import gym
from big_guys.algos.core import Buffer

def VPG(env_name, actor_critic=core.MLPActorCritic, ac_kwargs=dict(), seed=0,
        steps_per_epoch=4000, epochs=50, gamma=0.99, pi_lr=3e-4,
        vf_lr=1e-3, train_v_iters=80, lam=0.97, max_ep_len=1000,
        logger_kwargs=dict(), save_freq=10):
    torch.manual_seed(seed)
    np.random.seed(seed)

    env = gym.make(env_name)
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape

    ac = actor_critic(env.observation_space, env.action_space, **ac_kwargs)

    buf = Buffer(obs_dim, act_dim, steps_per_epoch, gamma, lam)

    def compute_loss_pi(data):
        obs, act, adv, logp_old = data['obs'], data['act'], data['adv'], data['logp']

        pi, logp = ac.pi(obs, act)
        loss_pi = -(logp * adv).mean()

        approx_kl = (logp_old - logp).mean().item()
        ent = pi.entropy().mean().item()
        pi_info = dict(kl=approx_kl, ent=ent)

        return loss_pi, pi_info

    def compute_loss_v(data):
        obs, ret = data["obs"], data['ret']
        return ((ac.v(obs) - ret) ** 2).mean()

    pi_opt = Adam(ac.pi.parameters(), lr=pi_lr)
    vf_opt = Adam(ac.v.parameters(), lr=vf_lr)

    def update():
        data = buf.get()  # trajectory info

        pi_l_old, pi_info_old = compute_loss_pi(data)
        pi_l_old = pi_l_old.item()
        v_l_old = compute_loss_v(data).item()

        pi_opt.zero_grad()
        loss_pi, pi_info = compute_loss_pi(data)
        loss_pi.backward()
        pi_opt.step()

        for i in range(train_v_iters):
            vf_opt.zero_grad()
            loss_v = compute_loss_v(data)
            loss_v.backward()
            vf_opt.step()

    o, ep_ret, ep_len = env.reset(), 0, 0
    for epoch in range(epochs):
        for t in range(steps_per_epoch):
            a, v, logp = ac.step(torch.as_tensor(o, dtype=torch.float32))

            next_o, r, d, _ = env.step(a)
            ep_ret += r
            ep_len += 1

            buf.store(o, a, r, v, logp)

            o = next_o

            timeout = ep_len == max_ep_len
            terminal = d or timeout
            epoch_ended = t == steps_per_epoch - 1

            if terminal or epoch_ended:
                if timeout or epoch_ended:
                    _, v, _ = ac.step(torch.as_tensor(o, dtype=torch.float32))
                else:
                    v = 0
                buf.finish_path(v)
                o, ep_ret, ep_len = env.reset(), 0, 0
        update()

VPG(env_name="CartPole-v0")