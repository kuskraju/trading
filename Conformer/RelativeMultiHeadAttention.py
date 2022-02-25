import copy

import torch
import torch.nn as nn
import torch.nn.functional as f
import math
import numpy as np


def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def attention(q, k, v, d_k, rel_pos, dropout=None):
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)

    scores += torch.einsum('bhad,adl->bhal', q, rel_pos)

    scores = F.softmax(scores, dim=-1)
    if dropout is not None:
        scores = dropout(scores)
    output = torch.matmul(scores, v)
    return output

class RelativeMultiHeadAttention(nn.Module):

    def __init__(self, device, heads, d_model, length, dropout=0.1):
        super().__init__()
        self.rel_pos = None
        self.device = device

        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads
        self.length = length

        self.q_linear = nn.Linear(d_model, d_model).to(self.device)
        self.v_linear = nn.Linear(d_model, d_model).to(self.device)
        self.k_linear = nn.Linear(d_model, d_model).to(self.device)
        self.dropout = nn.Dropout(dropout).to(self.device)
        self.out = nn.Linear(d_model, d_model).to(self.device)

        self.w = torch.nn.parameter.Parameter(torch.zeros(2 * self.length - 1, self.d_k), requires_grad=True)

    def forward(self, q, k, v):
        bs = q.size(0)

        # perform linear operation and split into N heads
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)

        # transpose to get dimensions bs * N * sl * d_model
        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)

        rel_pos = self.w.unfold(0, self.length, 1).to(self.device)

        # calculate attention using function we will define next
        scores = self._attention(q, k, v, self.d_k, rel_pos, self.dropout)
        # concatenate heads and put through final linear layer
        concat = scores.transpose(1, 2).contiguous().view(bs, -1, self.d_model)
        output = self.dropout(self.out(concat))
        return output
