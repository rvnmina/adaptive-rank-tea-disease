import math
import torch
import torch.nn as nn
IMG_SIZE, PATCH, D, DEPTH, MLP_DIM, DROPOUT = 96, 16, 64, 4, 128, 0.1
N_TOKENS = (IMG_SIZE // PATCH) ** 2 + 1
W_SCALE = 0.1
class RandomFeatureMap(nn.Module):
    def __init__(self, d, r, w_scale=W_SCALE, seed=42):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.register_buffer("W", torch.randn(r, d, generator=g) * w_scale)
        self.register_buffer("b", torch.rand(r, generator=g) * 2 * math.pi)
        m = torch.arange(1, r + 1, dtype=torch.float32)
        self.register_buffer("m_over_r", m / r)
    def forward(self, x):
        proj = x @ self.W.t()
        return torch.cos(2.0 * math.pi * self.m_over_r * proj + self.b)
class P3Attention(nn.Module):
    def __init__(self, d, r, dropout=DROPOUT, seed=42):
        super().__init__()
        self.W_q = nn.Linear(d, d, bias=False)
        self.W_k = nn.Linear(d, d, bias=False)
        self.W_v = nn.Linear(d, d, bias=False)
        self.W_o = nn.Linear(d, d, bias=False)
        for lin in (self.W_q, self.W_k, self.W_v, self.W_o):
            nn.init.xavier_uniform_(lin.weight)
        self.phi = RandomFeatureMap(d, r, seed=seed)
        self.drop = nn.Dropout(dropout)
    def forward(self, x, r=None):
        Q, K, V = self.W_q(x), self.W_k(x), self.W_v(x)
        Phi_Q, Phi_K = self.phi(Q), self.phi(K)
        KtV = torch.bmm(Phi_K.transpose(1, 2), V)
        Z = torch.bmm(Phi_Q, KtV)
        return self.drop(self.W_o(Z))
class SoftmaxAttention(nn.Module):
    def __init__(self, d, dropout=DROPOUT):
        super().__init__()
        self.W_q = nn.Linear(d, d, bias=False)
        self.W_k = nn.Linear(d, d, bias=False)
        self.W_v = nn.Linear(d, d, bias=False)
        self.W_o = nn.Linear(d, d, bias=False)
        for lin in (self.W_q, self.W_k, self.W_v, self.W_o):
            nn.init.xavier_uniform_(lin.weight)
        self.scale = d ** -0.5
        self.drop = nn.Dropout(dropout)
    def attention_map(self, x):
        Q, K = self.W_q(x), self.W_k(x)
        return torch.softmax(Q @ K.transpose(-2, -1) * self.scale, dim=-1)
    def forward(self, x, r=None):
        Z = self.attention_map(x) @ self.W_v(x)
        return self.drop(self.W_o(Z))
class Block(nn.Module):
    def __init__(self, attn, d, mlp_dim, dropout=DROPOUT):
        super().__init__()
        self.norm1 = nn.LayerNorm(d)
        self.attn = attn
        self.norm2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(
            nn.Linear(d, mlp_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(mlp_dim, d), nn.Dropout(dropout),
        )
    def forward(self, x, r=None):
        x = x + self.attn(self.norm1(x), r)
        x = x + self.mlp(self.norm2(x))
        return x
class ViTBase(nn.Module):
    def __init__(self, make_attn, img_size, patch_size, in_chans, num_classes, d, depth, mlp_dim, dropout):
        super().__init__()
        self.patch_size = patch_size
        self.d = d
        num_patches = (img_size // patch_size) ** 2
        self.patch_embed = nn.Linear(in_chans * patch_size ** 2, d)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, d))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.pos_drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([Block(make_attn(i), d, mlp_dim, dropout) for i in range(depth)])
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, num_classes)
        nn.init.normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)
    def embed(self, x):
        B, C, p = x.shape[0], x.shape[1], self.patch_size
        x = x.unfold(2, p, p).unfold(3, p, p)
        x = x.permute(0, 2, 3, 1, 4, 5).reshape(B, -1, C * p * p)
        x = self.patch_embed(x)
        x = torch.cat([self.cls_token.expand(B, -1, -1), x], dim=1)
        return self.pos_drop(x + self.pos_embed)
    def forward(self, x, r=None):
        x = self.embed(x)
        ranks = r if isinstance(r, (list, tuple)) else [r] * len(self.blocks)
        for blk, rk in zip(self.blocks, ranks):
            x = blk(x, rk)
        return self.head(self.norm(x[:, 0]))
class P3ViT(ViTBase):
    def __init__(self, img_size=IMG_SIZE, patch_size=PATCH, in_chans=3, num_classes=6, d=D, depth=DEPTH, mlp_dim=MLP_DIM, r=32, dropout=DROPOUT):
        super().__init__(lambda i: P3Attention(d, r, dropout, seed=42 + i), img_size, patch_size, in_chans, num_classes, d, depth, mlp_dim, dropout)
        self.r = r
class SoftmaxViT(ViTBase):
    def __init__(self, img_size=IMG_SIZE, patch_size=PATCH, in_chans=3, num_classes=6, d=D, depth=DEPTH, mlp_dim=MLP_DIM, dropout=DROPOUT):
        super().__init__(lambda i: SoftmaxAttention(d, dropout), img_size, patch_size, in_chans, num_classes, d, depth, mlp_dim, dropout)