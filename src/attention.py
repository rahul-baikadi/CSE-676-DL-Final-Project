import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, max_length, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        # One linear layer produces Q, K and V together.
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        # Lower-triangular mask: position t can see positions 0..t.
        mask = torch.tril(torch.ones(max_length, max_length, dtype=torch.bool))
        self.register_buffer("mask", mask.view(1, 1, max_length, max_length))

    def forward(self, x):
        # x: (B, T, d_model)
        B, T, C = x.shape

        q, k, v = self.qkv(x).split(C, dim=2)  # each (B, T, d_model)

        # Split into heads: (B, n_heads, T, d_head)
        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        # Attention scores: (B, n_heads, T, T)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
        scores = scores.masked_fill(~self.mask[:, :, :T, :T], float("-inf"))

        weights = F.softmax(scores, dim=-1)
        self.last_attn_weights = weights.detach()   # (B, n_heads, T, T), saved for inspection
        weights = self.attn_dropout(weights)


        out = weights @ v                                  # (B, n_heads, T, d_head)
        out = out.transpose(1, 2).contiguous().view(B, T, C)  # merge heads

        return self.resid_dropout(self.out_proj(out))
