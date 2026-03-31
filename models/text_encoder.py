"""models/text_encoder.py — Transformer text encoder + word-level tokenizer."""

import re, json
from collections import Counter
import torch
import torch.nn as nn
import torch.nn.functional as F


class Tokenizer:
    PAD, UNK, BOS, EOS = 0, 1, 2, 3

    def __init__(self, vocab_size=5000, max_len=32):
        self.vocab_size = vocab_size
        self.max_len    = max_len
        self.w2i = {"<PAD>": 0, "<UNK>": 1, "<BOS>": 2, "<EOS>": 3}
        self.i2w = {v: k for k, v in self.w2i.items()}

    def _tokenize(self, text):
        return re.sub(r"[^a-z0-9' ]+", " ", text.lower()).split()

    def build(self, sentences):
        counts = Counter(w for s in sentences for w in self._tokenize(s))
        for word, _ in counts.most_common(self.vocab_size - 4):
            idx = len(self.w2i)
            self.w2i[word] = idx
            self.i2w[idx]  = word

    def save(self, path):
        json.dump(self.w2i, open(path, "w"))

    def load(self, path):
        self.w2i = json.load(open(path))
        self.i2w = {int(v): k for k, v in self.w2i.items()}
        self.vocab_size = len(self.w2i)

    def encode(self, texts):
        """Encode a list of strings → (B, max_len) int64 tensor."""
        batch = []
        for text in texts:
            ids = [self.BOS] + [self.w2i.get(w, self.UNK) for w in self._tokenize(text)] + [self.EOS]
            ids = ids[:self.max_len]
            ids += [self.PAD] * (self.max_len - len(ids))
            batch.append(ids)
        return torch.tensor(batch, dtype=torch.long)

    def __len__(self):
        return len(self.w2i)


class TextTransformer(nn.Module):
    def __init__(self, vocab_size, max_len, embed_dim, num_heads, num_layers, proj_dim, dropout):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim * 4,
            dropout=dropout, batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.proj = nn.Linear(embed_dim, proj_dim)

    def forward(self, token_ids):
        pad_mask = (token_ids == 0)                             # True = ignore
        x = self.embed(token_ids)
        x = self.transformer(x, src_key_padding_mask=pad_mask)
        # Mean-pool over non-padding positions
        valid = (~pad_mask).float().unsqueeze(-1)
        x     = (x * valid).sum(1) / valid.sum(1).clamp(min=1e-8)
        return F.normalize(self.proj(x), dim=-1)                # (B, proj_dim) — L2-normalised
