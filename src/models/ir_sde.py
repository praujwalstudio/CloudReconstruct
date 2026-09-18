"""Mean-Reverting Inversion SDE (IR-SDE) for Dense Cloud Removal.

Based on the EMRDM (CVPR'25) formulation:
Forward process relaxes from clean target x_0 towards cloudy observation y:
    x_t = x_0 * mu(t) + y * (1 - mu(t)) + sigma(t) * epsilon

Reverse process starts from cloudy input y (at t=1) and reconstructs clean x_0 (at t=0)
conditioned on SAR, DEM, and optical context using fast ODE/SDE integration.
"""

import math
from typing import Callable, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


class IRSDE(nn.Module):
    """Mean-Reverting SDE (IR-SDE) for satellite optical cloud reconstruction."""

    def __init__(
        self,
        num_timesteps: int = 100,
        lambda_rate: float = 2.5,
        sigma_max: float = 0.5,
        sigma_min: float = 0.01,
        schedule: str = "cosine",
    ):
        super().__init__()
        self.num_timesteps = num_timesteps
        self.lambda_rate = float(lambda_rate)
        self.sigma_max = float(sigma_max)
        self.sigma_min = float(sigma_min)
        self.schedule = schedule

        # Precompute discrete schedules for fast lookup
        timesteps = torch.linspace(0.0, 1.0, num_timesteps + 1)
        mu_t, sigma_t = self._compute_schedule(timesteps)
        self.register_buffer("mu_table", mu_t)
        self.register_buffer("sigma_table", sigma_t)

    def _compute_schedule(self, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute mu(t) and sigma(t) for given normalized continuous time t in [0, 1]."""
        if self.schedule == "cosine":
            # Cosine decay schedule
            mu = torch.cos((t * math.pi) / 2.0).clamp(min=1e-5, max=1.0)
            sigma = self.sigma_max * torch.sin((t * math.pi) / 2.0).clamp(min=self.sigma_min)
        elif self.schedule == "exponential":
            # Exponential decay schedule
            mu = torch.exp(-self.lambda_rate * t).clamp(min=1e-5, max=1.0)
            sigma = self.sigma_max * torch.sqrt(1.0 - mu ** 2).clamp(min=self.sigma_min)
        else:
            # Linear decay schedule
            mu = (1.0 - t).clamp(min=1e-5, max=1.0)
            sigma = (self.sigma_min + (self.sigma_max - self.sigma_min) * t).clamp(min=self.sigma_min)
        return mu, sigma

    def get_schedule(self, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (mu_t, sigma_t) for continuous time tensor t in [0, 1]."""
        return self._compute_schedule(t)

    def q_sample(
        self,
        x_0: torch.Tensor,
        y: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward perturb: corrupt clear target x_0 towards cloudy y at time t.

        Args:
            x_0: Clear ground truth optical image (B, C, H, W)
            y: Cloudy optical image (B, C, H, W)
            t: Normalized time in [0, 1] (B,)
            noise: Optional Gaussian noise epsilon ~ N(0, I)

        Returns:
            Tuple of (x_t, noise)
        """
        if noise is None:
            noise = torch.randn_like(x_0)

        mu, sigma = self.get_schedule(t)
        mu = mu.view(-1, 1, 1, 1)
        sigma = sigma.view(-1, 1, 1, 1)

        # Mean-reverting interpolation: x_t = x_0 * mu + y * (1 - mu) + sigma * eps
        x_t = x_0 * mu + y * (1.0 - mu) + sigma * noise
        return x_t, noise

    @torch.no_grad()
    def sample_ode(
        self,
        model: Callable[..., torch.Tensor],
        y: torch.Tensor,
        sar: torch.Tensor,
        steps: int = 20,
        dem: Optional[torch.Tensor] = None,
        x_start_noise_scale: float = 0.8,
    ) -> torch.Tensor:
        """Fast deterministic ODE sampler (20-50 steps) for mean-reverting diffusion.

        Args:
            model: Neural network f(x_t, sar, t, dem) -> predicted clean optical x_0
            y: Cloudy optical image (B, C, H, W)
            sar: Sentinel-1 SAR image (B, C_sar, H, W)
            steps: Number of integration steps (e.g. 20)
            dem: Optional DEM elevation features
            x_start_noise_scale: Noise level to add to initial state at t=1

        Returns:
            Reconstructed clear optical image (B, C, H, W)
        """
        device = y.device
        batch_size = y.shape[0]

        # Initialize x_1 at t=1: cloudy input perturbed by noise
        sigma_1 = self.sigma_max * x_start_noise_scale
        x_t = y + sigma_1 * torch.randn_like(y)

        time_grid = torch.linspace(1.0, 0.0, steps + 1, device=device)

        for step_idx in range(steps):
            t_cur = time_grid[step_idx]
            t_next = time_grid[step_idx + 1]

            t_tensor = torch.full((batch_size,), t_cur.item(), device=device, dtype=torch.float32)

            # Model predicts clear target x_0 given current noisy optical state x_t
            pred_x0 = model(x_t, sar, t_tensor, dem=dem)

            # Compute current and next schedule parameters
            mu_cur, sigma_cur = self.get_schedule(t_cur)
            mu_next, sigma_next = self.get_schedule(t_next)

            # Inferred standardized noise direction
            noise_dir = (x_t - pred_x0 * mu_cur - y * (1.0 - mu_cur)) / (sigma_cur + 1e-7)

            # Deterministic ODE step to t_next
            if step_idx == steps - 1:
                # Final step reaches t=0 directly
                x_t = pred_x0
            else:
                x_t = pred_x0 * mu_next + y * (1.0 - mu_next) + sigma_next * noise_dir

        return torch.clamp(x_t, 0.0, 1.0)

    @torch.no_grad()
    def sample_sde(
        self,
        model: Callable[..., torch.Tensor],
        y: torch.Tensor,
        sar: torch.Tensor,
        steps: int = 50,
        dem: Optional[torch.Tensor] = None,
        eta: float = 0.5,
    ) -> torch.Tensor:
        """Stochastic reverse SDE sampler with Langevin diffusion dynamics."""
        device = y.device
        batch_size = y.shape[0]

        sigma_1 = self.sigma_max
        x_t = y + sigma_1 * torch.randn_like(y)

        time_grid = torch.linspace(1.0, 0.0, steps + 1, device=device)

        for step_idx in range(steps):
            t_cur = time_grid[step_idx]
            t_next = time_grid[step_idx + 1]

            t_tensor = torch.full((batch_size,), t_cur.item(), device=device, dtype=torch.float32)
            pred_x0 = model(x_t, sar, t_tensor, dem=dem)

            mu_cur, sigma_cur = self.get_schedule(t_cur)
            mu_next, sigma_next = self.get_schedule(t_next)

            noise_dir = (x_t - pred_x0 * mu_cur - y * (1.0 - mu_cur)) / (sigma_cur + 1e-7)

            if step_idx == steps - 1:
                x_t = pred_x0
            else:
                stoch_noise = torch.randn_like(x_t) * (eta * sigma_next)
                det_sigma = math.sqrt(max(1e-7, sigma_next.item() ** 2 - (eta * sigma_next.item()) ** 2))
                x_t = pred_x0 * mu_next + y * (1.0 - mu_next) + det_sigma * noise_dir + stoch_noise

        return torch.clamp(x_t, 0.0, 1.0)
