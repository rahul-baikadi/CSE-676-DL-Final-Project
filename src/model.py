import torch
import torch.nn as nn
import torch.nn.functional as F

from src.embeddings import TokenAndPositionEmbedding
from src.blocks import EncoderBlock, DecoderBlock


class Seq2SeqTransformer(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model=256,
        n_heads=4,
        n_enc_layers=2,
        n_dec_layers=2,
        max_src_len=32,
        max_tgt_len=256,
        dropout=0.1,
    ):
        super().__init__()
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len

        # Separate positional embeddings, one shared token embedding (Sec. 4.1)
        self.src_emb = TokenAndPositionEmbedding(vocab_size, d_model, max_src_len, dropout)
        self.tgt_emb = TokenAndPositionEmbedding(vocab_size, d_model, max_tgt_len, dropout)
        self.tgt_emb.token_embedding = self.src_emb.token_embedding

        self.encoder = nn.ModuleList(
            [EncoderBlock(d_model, n_heads, max_src_len, dropout) for _ in range(n_enc_layers)]
        )
        self.decoder = nn.ModuleList(
            [DecoderBlock(d_model, n_heads, max_tgt_len, dropout) for _ in range(n_dec_layers)]
        )

        # Pre-LN blocks need a final LayerNorm on each side
        self.ln_enc = nn.LayerNorm(d_model)
        self.ln_dec = nn.LayerNorm(d_model)

        # Output head, weights tied to the token embedding (Sec. 4.5)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.src_emb.token_embedding.weight

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    # ---------------- forward pieces ----------------
    def encode(self, src_ids, src_mask):
        h = self.src_emb(src_ids)
        for block in self.encoder:
            h = block(h, src_mask)
        return self.ln_enc(h)                       # (B, S, d_model)

    def decode(self, tgt_input, h_enc, src_mask, tgt_mask=None):
        h = self.tgt_emb(tgt_input)
        for block in self.decoder:
            h = block(h, h_enc, src_mask, tgt_mask)
        return self.ln_dec(h)                       # (B, T, d_model)  <- value head goes here later

    def forward(self, src_ids, src_mask, tgt_input, tgt_mask=None, labels=None):
        h_enc = self.encode(src_ids, src_mask)
        h_dec = self.decode(tgt_input, h_enc, src_mask, tgt_mask)
        logits = self.lm_head(h_dec)                # (B, T, vocab)

        loss = None
        if labels is not None:
            # Eq. 2: cross-entropy on story tokens only; padding is -100
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100
            )
        return logits, loss

    # ---------------- generation (Sec. 4.6) ----------------
    @torch.no_grad()
    def generate(self, src_ids, src_mask, eos_id, max_new_tokens=255,
                 temperature=1.0, top_k=50):
        self.eval()
        B = src_ids.size(0)
        h_enc = self.encode(src_ids, src_mask)      # encode once, reuse every step

        out = torch.full((B, 1), eos_id, dtype=torch.long, device=src_ids.device)
        finished = torch.zeros(B, dtype=torch.bool, device=src_ids.device)

        for _ in range(min(max_new_tokens, self.max_tgt_len - 1)):
            h_dec = self.decode(out, h_enc, src_mask)
            logits = self.lm_head(h_dec[:, -1, :]) / max(temperature, 1e-5)

            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = float("-inf")

            next_tok = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
            next_tok[finished] = eos_id             # keep finished rows padded with eos
            out = torch.cat([out, next_tok], dim=1)

            finished |= next_tok.squeeze(1) == eos_id
            if finished.all():
                break

        return out[:, 1:]                           # drop the start token
