import math
import torch
from torch import nn
import torch.nn.functional as f

from hidden_module.Conformer.ConformerEncoderLayer import ConformerEncoderLayer
from hidden_module.Conformer.RelativeMultiHeadAttention import get_clones
from hidden_module.initilizers import linear_init_with_he_normal, linear_init_with_zeros


class Encoder(nn.Module):
    def __init__(self, device, d_model, n, heads, dropout, features_count, length):
        super().__init__()
        self.device = device
        self.n = n
        self.d_model = d_model

        self.layers = get_clones(ConformerEncoderLayer(self.device, self.d_model, heads, length, dropout), n)
        self.norm = nn.LayerNorm(self.d_model).to(self.device)
        self.embed = nn.Linear(features_count, d_model).to(self.device)
        self.dropout = nn.Dropout(dropout).to(self.device)

    def forward(self, src):
        src = self.dropout(self.embed(src))
        for i in range(self.n):
            src = self.layers[i](src)
        return self.norm(src)


class ConformerOrdinal(nn.Module):

    def __init__(self, device, d_model, n, heads, dropout, features_count, length, classes):
        super().__init__()
        self.device = device
        self.classes = classes
        self.d_model = d_model
        self.d4 = self.d_model // 4

        self.encoder = Encoder(self.device, d_model, n, heads, dropout, features_count, length)

        self.out1 = linear_init_with_he_normal(
            nn.Linear(self.d_model * length, self.d4 * length)).to(device)
        self.out2 = linear_init_with_he_normal(
            nn.Linear(self.d4 * length, math.floor(math.sqrt(self.d4 * length * classes)))).to(self.device)
        self.out3 = linear_init_with_zeros(
            nn.Linear(math.floor(math.sqrt(self.d4 * length * classes)), 1)).to(device)
        self.dropout = nn.Dropout(dropout)

        step = 1 / classes
        self.classes = classes
        self.tresholds = ((torch.range(step, 1 - step / 2, step, requires_grad=False) - 0.5) * 2 * classes).to(self.device)

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.shape[0], -1)
        x = f.relu(self.out1(x))
        x = f.relu(self.out2(x))
        x = self.out3(x)
        x = torch.hstack((torch.sigmoid(self.tresholds.view(1, -1).repeat(x.shape[0], 1) - x),
                          torch.ones(x.shape[0], 1).to(self.device)))
        x_rest = x[..., 1:] - x[..., :-1]
        x = torch.hstack((x[..., 0].view(-1, 1), x_rest))
        return x
