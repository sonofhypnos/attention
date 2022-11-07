import numpy as np
import torch
import torch.nn as nn


class Attention(nn.Module):
    def __init__(self):
        super(Attention, self)

    def forward(self, Q, K, V):
        # FIXME not sure which is the right way to do the softmax (directly depends on how we arange V?)
        return torch.softmax(Q @ K.T, 0, torch.float) @ V
        # TODO nice to have would be scaling by sqrt(d_k) (supposedly helpful for gradients)


def run_single_attention():
    q = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    k = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    v = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    attention = Attention()
    print(attention.forward(q, k, v))


run_single_attention()
