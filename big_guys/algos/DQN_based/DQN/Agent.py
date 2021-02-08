import torch, random
import torch.optim as optim
import math
import torch.nn.functional as F
import numpy as np
from toss.architectures.SimpleCNN import SimpleCNN
from RL.big_guys.algos.DQN_based.DQN.DuelingCNN import DuelingCNN

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Takes step number and returns eps according to schedule
class EpsSchedule(object):
    def __init__(self, start, end, decay):
        self.start = start
        self.end = end
        self.decay = decay
        self.n_steps = 1
        self.epses = []

    def update(self):
        eps = self.end + (self.start - self.end) * math.exp(-1. * self.n_steps / self.decay)
        self.epses.append(eps)
        self.n_steps += 1
        return eps


class Agent:
    def __init__(self, obs_dim, n_actions, lr=1e-3, gamma=0.99, eps_sched= dict(),
                 double=False, dueling=False, n_filters=[8, 16, 32]):
        self.double = double

        self.gamma = gamma
        self.n_actions = n_actions

        self.losses = []
        self.eps_sched = EpsSchedule(**eps_sched)

        CNN = DuelingCNN if dueling else SimpleCNN
        self.Q = CNN(obs_dim, n_actions, n_filters).to(device)

        self.Q_target = CNN(obs_dim, n_actions, n_filters).to(device)
        self.update_target()
        # freeze parameters
        for p in self.Q_target.parameters():
            p.requires_grad = False

        self.opt = optim.Adam(self.Q.parameters(), lr=lr)

    @property
    def loss(self):
        return self.losses

    @property
    def eps(self):
        return self.eps_sched.epses

    def act(self, obs, eps=None):
        if eps is None:  # as in natural paper we schedule eps for policy
            assert isinstance(self.eps_sched, EpsSchedule)
            eps = self.eps_sched.update()

        if np.random.random() > eps:
            with torch.no_grad():
                return self.Q(obs).max(1)[1].view(1, 1)  # returns one number - argmax action
        else:
            return torch.tensor([[random.randrange(self.n_actions)]], dtype=torch.long)

    def update_main(self, batch):
        o, a, r, o2, d = batch["obs"], batch["act"], batch["rew"], batch["obs2"], batch["done"]

        q = self.Q(o).gather(1, a.unsqueeze(1)).squeeze()

        with torch.no_grad():
            out = self.Q_target(o2)

            if self.double:  # Double DQN
                actions = self.Q(o2).argmax(1).unsqueeze(1)  # select argmax actions according to main Q
                q_target = out.gather(1, actions).squeeze()  # take estimates of the actions according to Q target
            else:
                q_target = out.max(1)[0]

            backup = r + self.gamma * (1 - d) * q_target

        # Compute new priorities for transitions in the batch (is used for rank-based prioritized replay)
        changes = None
        if "heap_idx" in batch.keys():
            heap_idx = batch["heap_idx"]
            new_priorities = torch.abs(q - backup).detach()
            changes = zip(heap_idx, new_priorities)

        # Use Huber loss instead of MSE, since former is less sensitive to outliers
        loss_q = F.smooth_l1_loss(q, backup)

        self.opt.zero_grad()
        loss_q.backward()

        # Gradient clipping improves performance (paper - https://arxiv.org/pdf/1511.06581.pdf)
        torch.nn.utils.clip_grad_norm_(self.Q.parameters(), 10.)

        self.opt.step()
        self.losses.append(loss_q.item())

        return changes

    def update_target(self):
        self.Q_target.load_state_dict(self.Q.state_dict())  # Copy parameters
