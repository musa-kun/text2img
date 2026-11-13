"""Image generation: run the reverse diffusion loop with classifier-free guidance."""
from typing import List

import torch


@torch.no_grad()
def sample(
    model,
    scheduler,
    text_encoder,
    captions: List[str],
    image_size: int = 32,
    channels: int = 1,
    guidance_scale: float = 3.0,
    device: str = "cuda",
) -> torch.Tensor:
    """Generate one image per caption. Returns a tensor in [-1, 1], shape (B,C,H,W)."""
    was_training = model.training
    model.eval()

    b = len(captions)
    _, cond = text_encoder(captions)                 # (B, 512)
    _, uncond = text_encoder([""] * b)               # (B, 512) null caption

    x = torch.randn(b, channels, image_size, image_size, device=device)
    for t in reversed(range(scheduler.num_timesteps)):
        t_batch = torch.full((b,), t, device=device, dtype=torch.long)
        # One forward pass for both conditional and unconditional predictions.
        eps = model(
            torch.cat([x, x]), torch.cat([t_batch, t_batch]), torch.cat([cond, uncond])
        )
        eps_cond, eps_uncond = eps.chunk(2)
        eps = eps_uncond + guidance_scale * (eps_cond - eps_uncond)
        x = scheduler.step(eps, t, x)

    if was_training:
        model.train()
    return x.clamp(-1, 1)
