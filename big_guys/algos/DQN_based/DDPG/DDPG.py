import numpy as np
import gym, time, torch
from torch.optim import Adam
import RL.big_guys.algos.DQN_based.DDPG.core as core
from copy import deepcopy


class ReplayBuffer:
    def __init__(self, obs_dim, act_dim, size):
        self.obs_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.obs2_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done):
        self.obs_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(obs=self.obs_buf[idxs],
                     obs2=self.obs2_buf[idxs],
                     act=self.act_buf[idxs],
                     rew=self.rew_buf[idxs],
                     done=self.done_buf[idxs])
        return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in batch.items()}


def DDPG(env_name, actor_critic=core.MLPActorCritic, ac_kwargs=dict(), seed=0,
         steps_per_epoch=1000, epochs=1, replay_size=int(1e6), gamma=0.99,
         polyak=0.995, pi_lr=1e-3, q_lr=1e-3, batch_size=100, start_steps=1000,
         update_after=1000, update_every=50, act_noise=0.1, num_test_episodes=10,
         max_ep_len=1000, logger_kwargs=dict(), save_freq=1):
    # logger = EpochLogger(**logger_kwargs)
    #logger.save_config(locals())

    torch.manual_seed(seed)
    np.random.seed(seed)

    env, test_env = gym.make(env_name), gym.make(env_name)
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape[0] if env.action_space.shape else 1
    # act_limit = env.action_space.high[0]

    ac = actor_critic(env.observation_space, env.action_space, **ac_kwargs)
    ac_targ = deepcopy(ac)

    # Freeze target networks w.r.t optimizers (only update via polyak averaging)
    for p in ac_targ.parameters():
        p.requires_grad = False

    replay_buf = ReplayBuffer(obs_dim, act_dim, replay_size)

    def compute_loss_q(data):
        o, a, r, o2, d = data["obs"], data["act"], data["rew"], data["obs2"], data["done"]

        q = ac.q(o, a)

        with torch.no_grad():
            q_pi_targ = ac_targ.q(o2, ac_targ.pi(o2))
            backup = r + gamma * (1 - d) * q_pi_targ

        loss_q = ((q - backup) ** 2).mean()
        loss_info = dict(QVals=q.detach().numpy())

        return loss_q, loss_info

    def compute_loss_pi(data):
        o = data["obs"]
        q_pi = ac.q(o, ac.pi(o))
        return -q_pi.mean()

    pi_opt = Adam(ac.pi.parameters(), lr=pi_lr)
    q_opt = Adam(ac.q.parameters(), lr=q_lr)

    def update(data):
        q_opt.zero_grad()
        loss_q, loss_info = compute_loss_q(data)
        loss_q.backward()
        q_opt.step()

        for p in ac.q.parameters():
            p.requires_grad = False

        pi_opt.zero_grad()
        loss_pi = compute_loss_pi(data)
        loss_pi.backward()
        pi_opt.step()

        for p in ac.q.parameters():
            p.requires_grad = True

        #logger.store(LossQ=loss_q.item(), LossPi=loss_pi.item(), **loss_info)

        with torch.no_grad():
            for p, p_targ in zip(ac.parameters(), ac_targ.parameters()):
                p_targ.data.mul_(polyak)
                p_targ.data.add_((1 - polyak) * p.data)

    def get_action(o, noise_scale):
        a = ac.act(torch.as_tensor(o, dtype=torch.float32))
        a += noise_scale * np.random.randn(act_dim)
        # return np.clip(a, -act_limit, act_limit)
        return a

    def test_agent():
        for j in range(num_test_episodes):
            o, d, ep_ret, ep_len = test_env.reset(), False, 0, 0
            while not (d or (ep_len == max_ep_len)):
                o, r, d, _ = test_env.step(get_action(o, 0))
                ep_ret += r
                ep_len += 1
            # logger.store(TestEpRet=ep_ret, TestEpLen=ep_len)

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
            #logger.store(EpRet=ep_ret, EpLen=ep_len)
            o, ep_ret, ep_len = env.reset(), 0, 0

        if t >= update_after and t % update_every == 0:
            for _ in range(update_every):
                batch = replay_buf.sample_batch(batch_size)
                update(batch)

        if (t + 1) % steps_per_epoch == 0:
            epoch = (t+1) // steps_per_epoch

            test_agent()

            # logger.log_tabular('Epoch', epoch)
            # logger.log_tabular('EpRet', with_min_and_max=True)
            # logger.log_tabular('TestEpRet', with_min_and_max=True)
            # logger.log_tabular('EpLen', average_only=True)
            # logger.log_tabular('TestEpLen', average_only=True)
            # logger.log_tabular('TotalEnvInteracts', t)
            # logger.log_tabular('QVals', with_min_and_max=True)
            # logger.log_tabular('LossPi', average_only=True)
            # logger.log_tabular('LossQ', average_only=True)
            # logger.log_tabular('Time', time.time() - start_time)
            # logger.dump_tabular()

DDPG(env_name="Berzerk-v0")