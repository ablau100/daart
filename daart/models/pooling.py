# daart/models/pooling.py

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class MAB(nn.Module):
    def __init__(self, dim, num_heads, ln=False):
        super().__init__()
        self.dim_V = dim
        self.num_heads = num_heads
        self.fc_q = nn.Linear(dim, dim)
        self.fc_k = nn.Linear(dim, dim)
        self.fc_v = nn.Linear(dim, dim)
        if ln:
            self.ln0 = nn.LayerNorm(dim)
            self.ln1 = nn.LayerNorm(dim)
        self.fc_o = nn.Linear(dim, dim)

    def forward(self, Q, K):
        # Q, K: (B, S, dim)
        Q = self.fc_q(Q)
        K, V = self.fc_k(K), self.fc_v(K)
        d = self.dim_V // self.num_heads
        # split for multihead
        Q_ = torch.cat(Q.split(d, dim=2), dim=0)
        K_ = torch.cat(K.split(d, dim=2), dim=0)
        V_ = torch.cat(V.split(d, dim=2), dim=0)
        A = torch.softmax(Q_.bmm(K_.transpose(1,2)) / math.sqrt(d), dim=2)
        O = torch.cat((Q_ + A.bmm(V_)).split(Q.size(0), dim=0), dim=2)
        if hasattr(self, 'ln0'): O = self.ln0(O)
        O = O + F.relu(self.fc_o(O))
        if hasattr(self, 'ln1'): O = self.ln1(O)
        return O

class PMA(nn.Module):
    def __init__(self, dim, num_heads, num_seeds, ln=False):
        super().__init__()
        self.S = nn.Parameter(torch.Tensor(1, num_seeds, dim))
        nn.init.xavier_uniform_(self.S)
        self.mab = MAB(dim, num_heads, ln=ln)

    def forward(self, X):
        # X: (B, P, dim); returns (B, num_seeds, dim)
        B = X.size(0)
        return self.mab(self.S.repeat(B,1,1), X)
