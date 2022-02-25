import torch
import torch.nn as nn


class ConformerFeedForward(nn.Module):
    def __init__(self, device, d_model, dropout=0.1):
        super().__init__()
        self.device = device
        self.d_model = d_model
        self.norm = nn.LayerNorm(self.d_model).to(self.device)

        d_ff = d_model * 4

        self.linear_1 = nn.Linear(d_model, d_ff).to(self.device)
        self.linear_2 = nn.Linear(d_ff, d_model).to(self.device)

        self.dropout_1 = nn.Dropout(dropout).to(self.device)
        self.dropout_2 = nn.Dropout(dropout).to(self.device)

    def forward(self, x):
        residual = x
        x = self.norm(x)
        x2 = self.linear_1(x)
        x3 = self.swish(x2)
        x4 = self.dropout_1(x3)
        x5 = self.linear_2(x4)
        x6 = self.dropout_2(x5)
        result = residual + x6
        return result

    @staticmethod
    def swish(x):
        return x * torch.sigmoid(x)
