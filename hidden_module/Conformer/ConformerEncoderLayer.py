import torch.nn.functional as f
from torch import nn

from hidden_module.Conformer.ConformerFeedForward import ConformerFeedForward
from hidden_module.Conformer.RelativeMultiHeadAttention import RelativeMultiHeadAttention


class ConformerEncoderLayer(nn.Module):

    def __init__(self, device, d_model, heads, length, dropout=0.1):
        super().__init__()
        self.device = device
        self.d_model = d_model
        self.norm = nn.LayerNorm(self.d_model).to(self.device)

        self.attn = RelativeMultiHeadAttention(self.device, heads, d_model, length, dropout=dropout)

        self.ff_1 = ConformerFeedForward(self.device, d_model, dropout=dropout)
        self.ff_2 = ConformerFeedForward(self.device, d_model, dropout=dropout)
        self.conv = ConformerConvModule(self.device, d_model)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x):
        x = x + 0.5 * self.ff_1(x)
        attn = self.attn(x, x, x)
        x = x + attn
        x = x + self.conv(x)
        x = x + 0.5 * self.ff_2(x)
        x = self.norm(x)
        return x


class ConformerConvModule(nn.Module):
    def __init__(self, device, dim, causal = False, expansion_factor = 2, kernel_size = 31, dropout = 0.):
        super().__init__()
        self.device = device
        inner_dim = dim * expansion_factor
        padding = calc_same_padding(kernel_size)

        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            Transpose((1, 2)),
            nn.Conv1d(dim, inner_dim * 2, 1).to(self.device),
            GLU(dim=1).to(self.device),
            DepthWiseConv1d(inner_dim, inner_dim, kernel_size = kernel_size, padding = padding).to(self.device),
            nn.BatchNorm1d(inner_dim).to(self.device) if not causal else nn.Identity(),
            Swish().to(self.device),
            nn.Conv1d(inner_dim, dim, 1).to(self.device),
            Transpose((1, 2)),
            nn.Dropout(dropout)
        ).to(self.device)

    def forward(self, x):
        return self.net(x)


def calc_same_padding(kernel_size):
    pad = kernel_size // 2
    return (pad, pad + (kernel_size + 1) % 2)


class Swish(nn.Module):
    def forward(self, x):
        return x * x.sigmoid()


class Transpose(nn.Module):
    def __init__(self, dims):
        super().__init__()
        assert len(dims) == 2, 'dims must be a tuple of two dimensions'
        self.dims = dims

    def forward(self, x):
        return x.transpose(*self.dims)


class GLU(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        out, gate = x.chunk(2, dim=self.dim)
        return out * gate.sigmoid()


class DepthWiseConv1d(nn.Module):
    def __init__(self, chan_in, chan_out, kernel_size, padding):
        super().__init__()
        self.padding = padding
        self.conv = nn.Conv1d(chan_in, chan_out, kernel_size, groups = chan_in)
        self.conv_out = nn.Conv1d(chan_out, chan_out, 1)

    def forward(self, x):
        x = f.pad(x, self.padding)
        x = self.conv(x)
        return self.conv_out(x)
