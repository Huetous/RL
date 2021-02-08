import math
from toss.layers import Flatten
import torch.nn as nn
from toss.architectures.SimpleCNN import SimpleCNN


# Since modules V and A both propagate their error to the last convolutional
# layer we rescale gradients (paper - https://arxiv.org/pdf/1511.06581.pdf)
def scale_gradients(module, grad_out, grad_in):
    return tuple(map(lambda grad: grad / math.sqrt(2.0), grad_out))


class DuelingCNN(nn.Module):
    def __init__(self, channels_in, channels_out, n_filters):
        super().__init__()
        self.body = SimpleCNN(channels_in, channels_out, n_filters, include_head=False)

        self.V = nn.Linear(n_filters[-1], 1)
        self.A = nn.Linear(n_filters[-1], channels_out)

        self.V.register_backward_hook(scale_gradients)
        self.A.register_backward_hook(scale_gradients)

    def forward(self, x):
        s = self.body(x)
        A = self.A(s)
        return self.V(s) + A - A.mean(1).unsqueeze(1)


