"""Frozen CLIP text encoder that turns caption strings into embeddings."""
from typing import List, Tuple

import torch
import torch.nn as nn
from transformers import CLIPTextModel, CLIPTokenizer


class FrozenCLIPTextEncoder(nn.Module):
    """Wraps a pre-trained CLIP text tower. No gradients, always in eval mode.

    forward(captions) returns:
        seq_emb:    (B, max_length, embed_dim)  per-token embeddings
        pooled_emb: (B, embed_dim)              one vector per caption
    """

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        max_length: int = 16,
    ):
        super().__init__()
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
        self.model = CLIPTextModel.from_pretrained(model_name)
        self.max_length = max_length
        self.embed_dim = self.model.config.hidden_size  # 512 for this CLIP

        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    def train(self, mode: bool = True):
        # Ignore requests to enter training mode: the encoder stays frozen.
        return super().train(False)

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    @torch.no_grad()
    def forward(self, captions: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        tokens = self.tokenizer(
            list(captions),
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(self.device)
        out = self.model(**tokens)
        return out.last_hidden_state, out.pooler_output
