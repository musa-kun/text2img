"""Train the text-conditioned diffusion model on captioned Fashion-MNIST.

Run from the repo root:  python -m src.train --epochs 30
"""
import argparse
import json
import os
import random
import time

import torch
import torch.nn.functional as F

from src.dataset import CLASS_NAMES, get_dataloader
from src.model import MiniUNet
from src.sampling import sample
from src.scheduler import DDPMScheduler
from src.text_encoder import FrozenCLIPTextEncoder
from src.utils import (
    EMA, load_checkpoint, plot_loss, save_checkpoint, save_sample_grid, set_seed,
)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Train a tiny text-to-image diffusion model")
    p.add_argument("--data_dir", default="./data")
    p.add_argument("--out_dir", default="./outputs")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--ema_decay", type=float, default=0.999)
    p.add_argument("--p_uncond", type=float, default=0.1, help="caption dropout probability")
    p.add_argument("--guidance_scale", type=float, default=3.0)
    p.add_argument("--sample_every", type=int, default=5, help="epochs between sample grids")
    p.add_argument("--log_every", type=int, default=200, help="steps between loss logs")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume", action="store_true")
    return p.parse_args(argv)


def generate_grid(ema_model, scheduler, text_encoder, cfg, device, path):
    """Two rows of samples, one column per class, always from the same start noise."""
    prompts = [f"a photo of a {c}" for c in CLASS_NAMES] * 2
    devices = [torch.cuda.current_device()] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):   # don't disturb training's RNG
        torch.manual_seed(1234)
        images = sample(
            ema_model, scheduler, text_encoder, prompts,
            guidance_scale=cfg.guidance_scale, device=str(device),
        )
    save_sample_grid(images, path, nrow=len(CLASS_NAMES))


def main(argv=None):
    cfg = parse_args(argv)
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_dir = os.path.join(cfg.out_dir, "checkpoints")
    sample_dir = os.path.join(cfg.out_dir, "samples")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(sample_dir, exist_ok=True)

    loader = get_dataloader(root=cfg.data_dir, train=True, batch_size=cfg.batch_size)
    text_encoder = FrozenCLIPTextEncoder().to(device)
    model = MiniUNet(text_dim=text_encoder.embed_dim).to(device)
    scheduler = DDPMScheduler().to(device)
    ema = EMA(model, cfg.ema_decay)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    step, completed, history = 0, 0, []
    latest = os.path.join(ckpt_dir, "latest.pt")
    if cfg.resume and os.path.exists(latest):
        step, completed, history = load_checkpoint(latest, model, ema, optimizer, device)
        print(f"Resumed from epoch {completed} (step {step})")

    print(f"Device: {device} | trainable params: "
          f"{sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.2f}M")

    running, count = 0.0, 0
    for epoch in range(completed, cfg.epochs):
        model.train()
        t0 = time.time()
        epoch_loss, epoch_batches = 0.0, 0

        for images, captions in loader:
            images = images.to(device, non_blocking=True)

            # Caption dropout -> enables classifier-free guidance at sampling time.
            captions = ["" if random.random() < cfg.p_uncond else c for c in captions]
            _, text_emb = text_encoder(captions)

            t = torch.randint(0, scheduler.num_timesteps, (images.size(0),), device=device)
            x_t, noise = scheduler.add_noise(images, t)

            loss = F.mse_loss(model(x_t, t, text_emb), noise)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            ema.update(model)

            step += 1
            loss_val = loss.item()
            running += loss_val; count += 1
            epoch_loss += loss_val; epoch_batches += 1

            if step % cfg.log_every == 0:
                avg = running / count
                history.append({"step": step, "epoch": epoch + 1, "loss": avg})
                print(f"epoch {epoch + 1}/{cfg.epochs} | step {step} | loss {avg:.4f}")
                running, count = 0.0, 0

        print(f"== epoch {epoch + 1} done | mean loss {epoch_loss / epoch_batches:.4f} "
              f"| {time.time() - t0:.0f}s")

        save_checkpoint(latest, model, ema, optimizer, step, epoch + 1, history)

        is_last = epoch + 1 == cfg.epochs
        if epoch == 0 or (epoch + 1) % cfg.sample_every == 0 or is_last:
            generate_grid(ema.model, scheduler, text_encoder, cfg, device,
                          os.path.join(sample_dir, f"epoch_{epoch + 1:03d}.png"))
            save_checkpoint(os.path.join(ckpt_dir, f"ckpt_epoch{epoch + 1:03d}.pt"),
                            model, ema, optimizer, step, epoch + 1, history)
            print(f"   saved samples + checkpoint for epoch {epoch + 1}")

    with open(os.path.join(cfg.out_dir, "loss_history.json"), "w") as f:
        json.dump(history, f)
    if history:
        plot_loss(history, path=os.path.join(cfg.out_dir, "loss_curve.png"))


if __name__ == "__main__":
    main()
