import math
import torch
import torch.nn as nn
from p3_core import (ViTBase, IMG_SIZE, PATCH, D, DEPTH, MLP_DIM, DROPOUT, W_SCALE)
class NestedFeatureMap(nn.Module):
    def __init__(self, d, r_max=64, w_scale=W_SCALE, seed=42, learnable=False):
        super().__init__()
        self.r_max = r_max
        g = torch.Generator().manual_seed(seed)
        W = torch.randn(r_max, d, generator=g) * w_scale
        b = torch.rand(r_max, generator=g) * 2 * math.pi
        if learnable:
            self.W = nn.Parameter(W)
            self.b = nn.Parameter(b)
        else:
            self.register_buffer("W", W)
            self.register_buffer("b", b)
        m = torch.arange(1, r_max + 1, dtype=torch.float32)
        self.register_buffer("m_over_rmax", m / r_max)
    def forward(self, x, r=None):
        r = r or self.r_max
        proj = x @ self.W[:r].t()
        return torch.cos(2.0 * math.pi * self.m_over_rmax[:r] * proj + self.b[:r])
class NestedAttention(nn.Module):
    def __init__(self, d, r_max=64, dropout=DROPOUT, seed=42, learnable=False):
        super().__init__()
        self.W_q = nn.Linear(d, d, bias=False)
        self.W_k = nn.Linear(d, d, bias=False)
        self.W_v = nn.Linear(d, d, bias=False)
        self.W_o = nn.Linear(d, d, bias=False)
        for lin in (self.W_q, self.W_k, self.W_v, self.W_o):
            nn.init.xavier_uniform_(lin.weight)
        self.phi = NestedFeatureMap(d, r_max, seed=seed, learnable=learnable)
        self.drop = nn.Dropout(dropout)
    def forward(self, x, r=None):
        Q, K, V = self.W_q(x), self.W_k(x), self.W_v(x)
        Phi_Q, Phi_K = self.phi(Q, r), self.phi(K, r)
        KtV = torch.bmm(Phi_K.transpose(1, 2), V)
        Z = torch.bmm(Phi_Q, KtV)
        return self.drop(self.W_o(Z))
class NestedViT(ViTBase):
    def __init__(self, img_size=IMG_SIZE, patch_size=PATCH, in_chans=3, num_classes=6, d=D, depth=DEPTH, mlp_dim=MLP_DIM, r_max=64, dropout=DROPOUT, learnable=False):
        super().__init__(
            lambda i: NestedAttention(d, r_max, dropout, seed=42 + i, learnable=learnable),
            img_size, patch_size, in_chans, num_classes, d, depth, mlp_dim, dropout)
        self.r_max = r_max
        self.learnable = learnable