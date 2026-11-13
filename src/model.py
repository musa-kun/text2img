"""A small text- and time-conditioned U-Net for 32x32 diffusion."""
import math
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalTimeEmbedding(nn.Module):
    """Maps integer timesteps (B,) to sinusoidal features (B, dim)."""

    def __init__(self, dim: int):
        super().__init__()
        assert dim % 2 == 0, "dim must be even"
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=t.device) / half
        )
        args = t.float()[:, None] * freqs[None, :]
        return torch.cat([args.sin(), args.cos()], dim=-1)


class ResBlock(nn.Module):
    """Conv residual block, modulated by the conditioning vector (FiLM)."""

    def __init__(self, in_ch: int, out_ch: int, cond_dim: int, groups: int = 8):
        super().__init__()
        self.norm1 = nn.GroupNorm(groups, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.cond_proj = nn.Linear(cond_dim, out_ch * 2)  # -> scale and shift
        self.norm2 = nn.GroupNorm(groups, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        scale, shift = self.cond_proj(F.silu(cond))[:, :, None, None].chunk(2, dim=1)
        h = self.norm2(h) * (1 + scale) + shift
        h = self.conv2(F.silu(h))
        return h + self.skip(x)


class SelfAttention(nn.Module):
    """Spatial self-attention: every pixel attends to every other pixel."""

    def __init__(self, channels: int, heads: int = 4, groups: int = 8):
        super().__init__()
        self.norm = nn.GroupNorm(groups, channels)
        self.attn = nn.MultiheadAttention(channels, heads, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        tokens = self.norm(x).flatten(2).transpose(1, 2)        # (B, H*W, C)
        out, _ = self.attn(tokens, tokens, tokens, need_weights=False)
        return x + out.transpose(1, 2).reshape(b, c, h, w)


class Downsample(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x):
        return self.conv(F.interpolate(x, scale_factor=2, mode="nearest"))


class MiniUNet(nn.Module):
    """Predicts the noise in x_t given timestep t and a text embedding.

    Args:
        in_channels:   image channels (1 for Fashion-MNIST)
        base_channels: channels at the highest resolution
        channel_mults: per-level channel multipliers; len = number of levels
        attn_levels:   level indices that get self-attention
        text_dim:      size of the text embedding (512 for CLIP)
        cond_dim:      size of the shared conditioning vector
    """

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 64,
        channel_mults: Sequence[int] = (1, 2, 2),
        attn_levels: Sequence[int] = (2,),
        text_dim: int = 512,
        cond_dim: int = 256,
    ):
        super().__init__()
        chs = [base_channels * m for m in channel_mults]
        n = len(chs)

        # --- conditioning ---
        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(base_channels),
            nn.Linear(base_channels, cond_dim), nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )
        self.text_embed = nn.Sequential(
            nn.Linear(text_dim, cond_dim), nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )

        # --- encoder ---
        self.in_conv = nn.Conv2d(in_channels, chs[0], 3, padding=1)
        self.enc_blocks, self.enc_attns, self.downs = (
            nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        )
        prev = chs[0]
        for i, ch in enumerate(chs):
            self.enc_blocks.append(ResBlock(prev, ch, cond_dim))
            self.enc_attns.append(SelfAttention(ch) if i in attn_levels else nn.Identity())
            self.downs.append(Downsample(ch) if i < n - 1 else nn.Identity())
            prev = ch

        # --- middle ---
        self.mid1 = ResBlock(prev, prev, cond_dim)
        self.mid_attn = SelfAttention(prev)
        self.mid2 = ResBlock(prev, prev, cond_dim)

        # --- decoder (built from the deepest level back to the top) ---
        self.dec_blocks, self.dec_attns, self.ups = (
            nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        )
        for i in reversed(range(n)):
            self.dec_blocks.append(ResBlock(prev + chs[i], chs[i], cond_dim))  # + skip
            self.dec_attns.append(SelfAttention(chs[i]) if i in attn_levels else nn.Identity())
            self.ups.append(Upsample(chs[i]) if i > 0 else nn.Identity())
            prev = chs[i]

        # --- output ---
        self.out = nn.Sequential(
            nn.GroupNorm(8, chs[0]), nn.SiLU(),
            nn.Conv2d(chs[0], in_channels, 3, padding=1),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor, text_emb: torch.Tensor) -> torch.Tensor:
        """x: (B,C,H,W)   t: (B,) integer timesteps   text_emb: (B, text_dim)"""
        cond = self.time_embed(t) + self.text_embed(text_emb)

        h = self.in_conv(x)
        skips = []
        for block, attn, down in zip(self.enc_blocks, self.enc_attns, self.downs):
            h = attn(block(h, cond))
            skips.append(h)          # saved BEFORE downsampling
            h = down(h)

        h = self.mid2(self.mid_attn(self.mid1(h, cond)), cond)

        for block, attn, up in zip(self.dec_blocks, self.dec_attns, self.ups):
            h = torch.cat([h, skips.pop()], dim=1)   # skip connection
            h = attn(block(h, cond))
            h = up(h)

        return self.out(h)
