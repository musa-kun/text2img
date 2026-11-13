"""Generate images from trained weights.

Example:
    python -m src.generate --weights ema_weights.pt \
        --prompt "a photo of a sneaker" "a photo of a dress" --n 6 --out samples.png
"""
import argparse

import torch

from src.model import MiniUNet
from src.sampling import sample
from src.scheduler import DDPMScheduler
from src.text_encoder import FrozenCLIPTextEncoder
from src.utils import save_sample_grid, set_seed


def main(argv=None):
    p = argparse.ArgumentParser(description="Generate images from a trained model")
    p.add_argument("--weights", required=True, help="EMA weights (.pt) or a training checkpoint")
    p.add_argument("--prompt", nargs="+", default=["a photo of a sneaker"],
                   help="one or more quoted prompts")
    p.add_argument("--n", type=int, default=8, help="images per prompt")
    p.add_argument("--guidance_scale", type=float, default=3.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="generated.png")
    cfg = p.parse_args(argv)

    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    text_encoder = FrozenCLIPTextEncoder().to(device)
    scheduler = DDPMScheduler().to(device)
    model = MiniUNet(text_dim=text_encoder.embed_dim).to(device)

    state = torch.load(cfg.weights, map_location=device)
    if "ema" in state:            # a full training checkpoint
        state = state["ema"]
    model.load_state_dict(state)
    model.eval()

    prompts = [pr for pr in cfg.prompt for _ in range(cfg.n)]   # one row per prompt
    images = sample(model, scheduler, text_encoder, prompts,
                    guidance_scale=cfg.guidance_scale, device=str(device))
    save_sample_grid(images, cfg.out, nrow=cfg.n)
    print(f"Saved {len(prompts)} images to {cfg.out}")


if __name__ == "__main__":
    main()
