"""DDPM noise scheduler: forward noising and one reverse denoising step."""
import torch
import torch.nn as nn


class DDPMScheduler(nn.Module):
    """Holds the noise schedule as buffers (so .to(device) moves them too)."""

    def __init__(
        self,
        num_timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
    ):
        super().__init__()
        self.num_timesteps = num_timesteps

        betas = torch.linspace(beta_start, beta_end, num_timesteps)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        alpha_bars_prev = torch.cat([torch.ones(1), alpha_bars[:-1]])
        posterior_variance = betas * (1.0 - alpha_bars_prev) / (1.0 - alpha_bars)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)
        self.register_buffer("sqrt_alpha_bars", alpha_bars.sqrt())
        self.register_buffer("sqrt_one_minus_alpha_bars", (1.0 - alpha_bars).sqrt())
        self.register_buffer("posterior_variance", posterior_variance)

    @staticmethod
    def _gather(values: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Pick values[t] per sample and reshape to (B, 1, 1, 1) for broadcasting."""
        return values[t].reshape(-1, 1, 1, 1)

    def add_noise(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor = None):
        """Forward process. Returns (x_t, noise). t has shape (B,), dtype long."""
        if noise is None:
            noise = torch.randn_like(x0)
        x_t = (
            self._gather(self.sqrt_alpha_bars, t) * x0
            + self._gather(self.sqrt_one_minus_alpha_bars, t) * noise
        )
        return x_t, noise

    @torch.no_grad()
    def step(self, eps_pred: torch.Tensor, t: int, x_t: torch.Tensor) -> torch.Tensor:
        """One reverse step x_t -> x_{t-1}. t is a single Python int."""
        beta = self.betas[t]
        alpha = self.alphas[t]
        mean = (x_t - beta / self.sqrt_one_minus_alpha_bars[t] * eps_pred) / alpha.sqrt()
        if t == 0:
            return mean                       # no noise on the final step
        noise = torch.randn_like(x_t)
        return mean + self.posterior_variance[t].sqrt() * noise
