import numpy as np

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm(nn.Module):
    """Layer normalization."""

    def __init__(self, dim, eps=1e-3):
        super(LayerNorm, self).__init__()
        self.eps = eps
        self.a_2 = nn.Parameter(torch.ones(dim), requires_grad=True)
        self.b_2 = nn.Parameter(torch.zeros(dim), requires_grad=True)

    def forward(self, z):
        # if z.size(1) == 1:
        #     return z

        mu = torch.mean(z, keepdim=True, dim=-1)
        sigma = torch.std(z, keepdim=True, dim=-1)
        ln_out = (z - mu.expand_as(z)) / (sigma.expand_as(z) + self.eps)
        ln_out = ln_out * \
            self.a_2.expand_as(ln_out) + self.b_2.expand_as(ln_out)

        return ln_out
