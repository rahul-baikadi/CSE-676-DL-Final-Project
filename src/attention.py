import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):

    def __init__(self, d_model, n_heads, max_length, causal=False, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.causal = causal

        # Q comes from x; K and V come from the context (x itself for
        # self-attention, the encoder output for cross-attention).
        # Total parameters are still 4*d^2 + 4*d, the same as the fused qkv version.
        self.q_proj = nn.Linear(d_model, d_model)
        self.kv_proj = nn.Linear(d_model, 2 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        if causal:
            # Lower-triangular mask: position t can see positions 0..t.
            mask = torch.tril(torch.ones(max_length, max_length, dtype=torch.bool))
            self.register_buffer("causal_mask", mask.view(1, 1, max_length, max_length))

    def _split_heads(self, t):
        # (B, L, d_model) -> (B, n_heads, L, d_head)
        B, L, _ = t.shape
        return t.view(B, L, self.n_heads, self.d_head).transpose(1, 2)

    def forward(self, x, context=None, key_padding_mask=None):
        # x:       (B, T, d_model)  queries
        # context: (B, S, d_model)  keys/values; None means self-attention
        is_self = context is None
        if is_self:
            context = x

        B, T, C = x.shape
        S = context.shape[1]

        q = self._split_heads(self.q_proj(x))              # (B, h, T, d_head)
        k, v = self.kv_proj(context).split(C, dim=2)
        k = self._split_heads(k)                           # (B, h, S, d_head)
        v = self._split_heads(v)                           # (B, h, S, d_head)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)   # (B, h, T, S)

        if self.causal:
            assert is_self, "causal masking only applies to self-attention"
            scores = scores.masked_fill(~self.causal_mask[:, :, :T, :S], float("-inf"))

        if key_padding_mask is not None:
            # (B, S) -> (B, 1, 1, S): mask out padded key positions for every query
            keep = key_padding_mask.bool()[:, None, None, :]
            scores = scores.masked_fill(~keep, float("-inf"))

        weights = F.softmax(scores, dim=-1)
        self.last_attn_weights = weights.detach()   # saved for heatmaps
        weights = self.attn_dropout(weights)

        out = weights @ v                                       # (B, h, T, d_head)
        out = out.transpose(1, 2).contiguous().view(B, T, C)    # merge heads

        return self.resid_dropout(self.out_proj(out))


class CausalSelfAttention(MultiHeadAttention):

    def __init__(self, d_model, n_heads, max_length, dropout=0.1):
        super().__init__(d_model, n_heads, max_length, causal=True, dropout=dropout)
