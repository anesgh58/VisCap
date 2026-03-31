"""models/model.py — Multimodal model: VisionTransformer + TextTransformer + InfoNCE loss."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.vit_encoder  import VisionTransformer
from models.text_encoder import TextTransformer


class MultimodalModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.image_encoder = VisionTransformer(
            image_size=cfg["image_size"],
            patch_size=cfg["patch_size"],
            embed_dim=cfg["embed_dim"],
            num_heads=cfg["num_heads"],
            num_layers=cfg["num_layers"],
            proj_dim=cfg["proj_dim"],
            dropout=cfg["dropout"],
        )
        self.text_encoder = TextTransformer(
            vocab_size=cfg["vocab_size"],
            max_len=cfg["max_seq_len"],
            embed_dim=cfg["embed_dim"],
            num_heads=cfg["num_heads"],
            num_layers=cfg["num_layers"],
            proj_dim=cfg["proj_dim"],
            dropout=cfg["dropout"],
        )
        # Learnable temperature — initialised to cfg value, trained alongside the model
        self.log_temp = nn.Parameter(torch.tensor(cfg["temperature"]).log())

    @property
    def temperature(self):
        # Clamp to a safe range so it never collapses or explodes
        return self.log_temp.exp().clamp(min=0.01, max=1.0)

    def encode_image(self, images):
        """(B, 3, H, W) → (B, proj_dim) L2-normalised."""
        return self.image_encoder(images)

    def encode_text(self, token_ids):
        """(B, max_len) → (B, proj_dim) L2-normalised."""
        return self.text_encoder(token_ids)

    def forward(self, images, token_ids):
        """
        InfoNCE (CLIP-style) contrastive loss.

        Both encoders produce L2-normalised embeddings so the dot product
        equals cosine similarity.  The loss maximises similarity for the
        N diagonal (matching) pairs and minimises it for the N²-N
        off-diagonal (non-matching) pairs, symmetrically in both directions.
        """
        img_emb = self.encode_image(images)    # (B, D)
        txt_emb = self.encode_text(token_ids)  # (B, D)

        # Scaled cosine-similarity matrix
        logits  = (img_emb @ txt_emb.T) / self.temperature   # (B, B)
        targets = torch.arange(len(images), device=images.device)

        # Symmetric cross-entropy: average image→text and text→image directions
        loss_i2t = F.cross_entropy(logits,   targets)
        loss_t2i = F.cross_entropy(logits.T, targets)
        loss     = (loss_i2t + loss_t2i) / 2

        return loss, img_emb, txt_emb

    @torch.no_grad()
    def zero_shot(self, image, prompt_ids):
        """
        Return softmax probabilities over K text prompts for a single image.
        image      : (1, 3, H, W)
        prompt_ids : (K, max_len)
        returns    : (K,) probability tensor
        """
        img_emb = self.encode_image(image)           # (1, D)
        txt_emb = self.encode_text(prompt_ids)        # (K, D)
        logits  = (img_emb @ txt_emb.T) / self.temperature
        return F.softmax(logits, dim=-1).squeeze(0)   # (K,)
