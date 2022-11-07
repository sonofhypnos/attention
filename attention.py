import numpy as np
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

# TODO figure out embedding things


class Attention(nn.Module):
    def __init__(self):
        super(Attention, self)

    def forward(self, Q, K, V):
        # FIXME not sure which is the right way to do the softmax (directly depends on how we arange V?)
        # FIXME only works for quadratic matrices now
        return torch.softmax(Q @ K.T, 0, torch.float) @ V
        # TODO nice to have would be scaling by sqrt(d_k) (supposedly helpful for gradients)


class Attention_tunable(nn.Module):
    def __init__(self, d_k, d_v, d_model):
        super(Attention, self)
        self.w_q = Parameter(torch.empty(d_k, d_model))
        self.w_k = Parameter(torch.empty(d_k, d_model))
        self.w_v = Parameter(torch.empty(d_v, d_model))

    def forward(self, Q, K, V):
        """Allowed myself to look at the torch linear layer to figure out how to do this."""
        # FIXME not sure which is the right way to do the softmax (directly depends on how we arange V?)
        # FIXME only works for quadratic matrices now
        return torch.softmax((Q @ w_q) @ ((K @ w_k).T), 0, torch.float) @ (V @ w_v)
        # TODO nice to have would be scaling by sqrt(d_k) (supposedly helpful for gradients)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_k, dv, d_model, h=4):
        super(MultiHeadAttention, self)
        # we are making a very small attention head by default. Not sure the dimensions make any sense
        # TODO do matrix-mult/tensor-dot instead of for-loop:
        # FIXME figure out if the h_s stuff here will still work
        self.h_s = []

    def forward(self, Q, K, V):
        pass


def run_single_attention():
    q = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    k = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    v = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    attention = Attention()
    return attention.forward(q, k, v)


run_single_attention()
