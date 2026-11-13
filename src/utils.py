"""Training helpers: seeding, EMA, checkpoints, image grids, loss plots."""
import copy
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.utils import save_image


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class EMA:
    """Exponential moving average of a model's weights (kept as a separate model)."""

    def __init__(self, model: torch.nn.Module, decay: float = 0.999):
        self.decay = decay
        self.num_updates = 0
        self.model = copy.deepcopy(model).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        self.num_updates += 1
        d = min(self.decay, (1 + self.num_updates) / (10 + self.num_updates))
        for ema_p, p in zip(self.model.parameters(), model.parameters()):
            ema_p.mul_(d).add_(p.detach(), alpha=1 - d)


def save_checkpoint(path, model, ema, optimizer, step, completed_epochs, history):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    torch.save(
        {
            "model": model.state_dict(),
            "ema": ema.model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
            "completed_epochs": completed_epochs,
            "history": history,
        },
        tmp,
    )
    os.replace(tmp, path)  # atomic: an interrupted save can't corrupt the old file


def load_checkpoint(path, model, ema, optimizer, device):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"])
    ema.model.load_state_dict(ckpt["ema"])
    optimizer.load_state_dict(ckpt["optimizer"])
    ema.num_updates = ckpt["step"]
    return ckpt["step"], ckpt["completed_epochs"], ckpt["history"]


def save_sample_grid(images: torch.Tensor, path: str, nrow: int = 10, scale: int = 4) -> None:
    """Save images in [-1, 1] as an upscaled PNG grid."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    images = (images.detach().cpu() + 1) / 2
    images = F.interpolate(images, scale_factor=scale, mode="nearest")
    save_image(images, path, nrow=nrow, padding=2)


def plot_loss(history, path=None) -> None:
    import matplotlib.pyplot as plt

    steps = [h["step"] for h in history]
    losses = [h["loss"] for h in history]
    plt.figure(figsize=(7, 4))
    plt.plot(steps, losses)
    plt.yscale("log")
    plt.xlabel("step")
    plt.ylabel("training loss (MSE)")
    plt.title("Training loss")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    if path:
        plt.savefig(path, dpi=150)
    plt.show()
