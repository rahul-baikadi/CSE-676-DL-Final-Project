import torch
import torch.nn as nn


class TokenAndPositionEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model, max_length, dropout=0.1):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_length, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids):
        # input_ids: (batch_size, seq_len)
        seq_len = input_ids.size(1)
        positions = torch.arange(seq_len, device=input_ids.device)

        tok = self.token_embedding(input_ids)      # (B, T, d_model)
        pos = self.position_embedding(positions)   # (T, d_model), broadcast over B

        return self.dropout(tok + pos)
