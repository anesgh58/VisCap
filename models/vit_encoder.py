"""models/vit_encoder.py — Vision Transformer image encoder."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchEmbed(nn.Module):
    def __init__(self, image_size, patch_size, embed_dim):
        super().__init__()
        self.num_patches = (image_size // patch_size) ** 2
        self.proj = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)           # (B, D, H/P, W/P)
        x = x.flatten(2)           # (B, D, N)
        return x.transpose(1, 2)   # (B, N, D)


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn  = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn   = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        n = self.norm1(x)
        y, _ = self.attn(n, n, n)
        x = x + y
        x = x + self.ffn(self.norm2(x))
        return x


class VisionTransformer(nn.Module):
    def __init__(self, image_size, patch_size, embed_dim, num_heads, num_layers, proj_dim, dropout):
        super().__init__()
        self.patch_embed = PatchEmbed(image_size, patch_size, embed_dim)
        n = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, n + 1, embed_dim))
        self.blocks    = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, dropout) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.proj = nn.Linear(embed_dim, proj_dim)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):
        B = x.shape[0]
        x   = self.patch_embed(x)
        cls = self.cls_token.expand(B, -1, -1)
        x   = torch.cat([cls, x], dim=1) + self.pos_embed
        for block in self.blocks:
            x = block(x)
        x         = self.norm(x)
        cls_out   = x[:, 0]                              # (B, D)
        projected = F.normalize(self.proj(cls_out), dim=-1)
        return projected                                  # (B, proj_dim) — L2-normalised
