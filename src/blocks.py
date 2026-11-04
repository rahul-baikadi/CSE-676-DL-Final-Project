import torch.nn as nn

from src.attention import MultiHeadAttention


class FeedForward(nn.Module):
    """FFN(h) = W2 GELU(W1 h + b1) + b2, hidden size 4*d_model (Eq. 13)."""

    def __init__(self, d_model, dropout=0.1, mult=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, mult * d_model),
            nn.GELU(),
            nn.Linear(mult * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class EncoderBlock(nn.Module):
    """
    h'  = h  + SelfAttn_bi(LN(h))     (Eq. 14)
    h'' = h' + FFN(LN(h'))            (Eq. 15)
    """

    def __init__(self, d_model, n_heads, max_length, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.self_attn = MultiHeadAttention(d_model, n_heads, max_length,
                                            causal=False, dropout=dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, dropout)

    def forward(self, x, src_mask=None):
        # x: (B, S, d_model), src_mask: (B, S) 1 = real, 0 = pad
        x = x + self.self_attn(self.ln1(x), key_padding_mask=src_mask)
        x = x + self.ffn(self.ln2(x))
        return x


class DecoderBlock(nn.Module):
    """
    h'   = h   + SelfAttn_causal(LN(h))          (Eq. 16)
    h''  = h'  + CrossAttn(LN(h'), H_enc)        (Eq. 17)
    h''' = h'' + FFN(LN(h''))                    (Eq. 18)
    """

    def __init__(self, d_model, n_heads, max_length, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.self_attn = MultiHeadAttention(d_model, n_heads, max_length,
                                            causal=True, dropout=dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, max_length,
                                             causal=False, dropout=dropout)
        self.ln3 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, dropout)

    def forward(self, x, h_enc, src_mask=None, tgt_mask=None):
        # x: (B, T, d_model) decoder states
        # h_enc: (B, S, d_model) encoder output
        # tgt_mask is optional: with right-padding the causal mask already
        # keeps real tokens from seeing padding, and padded positions are -100 in the loss.
        x = x + self.self_attn(self.ln1(x), key_padding_mask=tgt_mask)
        x = x + self.cross_attn(self.ln2(x), context=h_enc, key_padding_mask=src_mask)
        x = x + self.ffn(self.ln3(x))
        return x
