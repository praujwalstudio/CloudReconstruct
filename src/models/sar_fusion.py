import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Down(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_ch, out_ch),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mpconv(x)


class Up(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]
        x1 = F.pad(
            x1, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2]
        )
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int = 32):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device, dtype=torch.float32) * -emb)
        emb = t.float().unsqueeze(-1) * emb.unsqueeze(0)
        return torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)


class CrossAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 4, dropout: float = 0.0):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.to_q = nn.Conv2d(dim, dim, 1)
        self.to_k = nn.Conv2d(dim, dim, 1)
        self.to_v = nn.Conv2d(dim, dim, 1)
        self.to_out = nn.Conv2d(dim, dim, 1)

    def forward(self, x: torch.Tensor, context: torch.Tensor = None) -> torch.Tensor:
        context = context or x
        B, C, H, W = x.shape
        q = self.to_q(x).view(B, self.num_heads, C // self.num_heads, -1)
        k = self.to_k(context).view(B, self.num_heads, C // self.num_heads, -1)
        v = self.to_v(context).view(B, self.num_heads, C // self.num_heads, -1)
        attn = torch.softmax(q.transpose(-2, -1) @ k * self.scale, dim=-1)
        out = (v @ attn.transpose(-2, -1)).view(B, C, H, W)
        return self.to_out(out)


class AttentionBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int = 4):
        super().__init__()
        self.norm = nn.BatchNorm2d(dim)
        self.attn = CrossAttention(dim, num_heads)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.attn(self.norm(x))


class SARConditionalUNet(nn.Module):
    """SAR Conditional U-Net for dense cloud penetration with Early Topographic DEM Fusion."""

    def __init__(
        self,
        sar_channels: int = 2,
        liss4_channels: int = 3,
        out_channels: int = 3,
        features: list[int] = None,
        enable_time: bool = False,
        time_dim: int = 32,
        enable_attention: bool = False,
        dem_channels: int = 0,
    ):
        super().__init__()
        if features is None:
            features = [32, 64, 128, 256]
        in_channels = sar_channels + liss4_channels
        self.dem_channels = dem_channels

        self.inc = DoubleConv(in_channels, features[0])
        self.down1 = Down(features[0], features[1])

        # Early fusion projection at the 2nd contraction level if configured
        if dem_channels > 0:
            self.dem_fusion = nn.Sequential(
                nn.Conv2d(features[1] + dem_channels, features[1], kernel_size=1),
                nn.BatchNorm2d(features[1]),
                nn.ReLU(inplace=True),
            )
        else:
            self.dem_fusion = None

        self.down2 = Down(features[1], features[2])
        self.down3 = Down(features[2], features[3])
        self.down4 = Down(features[3], features[3])
        self.up1 = Up(features[3] + features[3], features[2])
        self.up2 = Up(features[2] + features[2], features[1])
        self.up3 = Up(features[1] + features[1], features[0])
        self.up4 = Up(features[0] + features[0], features[0])
        self.outc = nn.Conv2d(features[0], out_channels, 1)

        self.enable_time = enable_time
        if enable_time:
            self.time_mlp = nn.Sequential(
                SinusoidalPosEmb(time_dim),
                nn.Linear(time_dim, features[0]),
                nn.SiLU(),
                nn.Linear(features[0], features[0]),
            )

        self.enable_attention = enable_attention
        if enable_attention:
            self.bottleneck_attn = AttentionBlock(features[3])
            self.skip_attn = AttentionBlock(features[0])

    def forward(
        self,
        liss4: torch.Tensor,
        sar: torch.Tensor,
        t: torch.Tensor = None,
        dem: torch.Tensor = None,
    ) -> torch.Tensor:
        """Forward pass with optional DEM early fusion at second contraction level."""
        x = torch.cat([liss4, sar], dim=1)
        x1 = self.inc(x)
        if self.enable_time and t is not None:
            t_emb = self.time_mlp(t).unsqueeze(-1).unsqueeze(-1)
            x1 = x1 + t_emb
        x2 = self.down1(x1)

        # Early Topographic DEM Fusion
        if dem is not None:
            if dem.shape[1] == 1:
                from src.preprocessing.dem_features import compute_dem_feature_stack
                dem_feats = compute_dem_feature_stack(dem)
            else:
                dem_feats = dem

            if self.dem_channels > 0 and dem_feats.shape[1] > self.dem_channels:
                dem_feats = dem_feats[:, :self.dem_channels]

            if dem_feats.shape[2:] != x2.shape[2:]:
                dem_down = F.interpolate(dem_feats, size=(x2.size(2), x2.size(3)), mode="bilinear", align_corners=False)
            else:
                dem_down = dem_feats

            if self.dem_fusion is not None and dem_down.shape[1] == self.dem_channels:
                x2 = self.dem_fusion(torch.cat([x2, dem_down], dim=1))
            else:
                x2 = x2 + dem_down[:, :1].expand(-1, x2.size(1), -1, -1) * 0.05

        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        if self.enable_attention:
            x5 = self.bottleneck_attn(x5)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        if self.enable_attention:
            x = self.skip_attn(x)
        x = self.up4(x, x1)
        return self.outc(x)


def cosine_beta_schedule(steps: int, s: float = 0.008) -> torch.Tensor:
    t = torch.linspace(0, steps, steps + 1, dtype=torch.float64)
    f_t = torch.cos((t / steps + s) / (1 + s) * math.pi / 2) ** 2
    alphas_cumprod = f_t / f_t[0]
    betas = 1 - alphas_cumprod[1:] / alphas_cumprod[:-1]
    return torch.clamp(betas, max=0.999).float()


from src.models.ir_sde import IRSDE


class SARDiffusionWrapper(nn.Module):
    """SAR-Guided Diffusion Wrapper supporting Mean-Reverting IR-SDE & Fast ODE Sampling."""

    def __init__(
        self,
        sar_channels: int = 2,
        liss4_channels: int = 3,
        out_channels: int = 3,
        noise_steps: int = 100,
        dem_channels: int = 0,
        use_ir_sde: bool = True,
    ):
        super().__init__()
        self.noise_steps = noise_steps
        self.use_ir_sde = use_ir_sde
        self.unet = SARConditionalUNet(
            sar_channels,
            liss4_channels,
            out_channels,
            enable_time=True,
            dem_channels=dem_channels,
        )

        self.ir_sde = IRSDE(num_timesteps=noise_steps, schedule="cosine")

        betas = cosine_beta_schedule(noise_steps)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", 1.0 - betas)
        self.register_buffer("alpha_bars", torch.cumprod(1.0 - betas, dim=0))

    @torch.no_grad()
    def forward(
        self,
        liss4: torch.Tensor,
        sar: torch.Tensor,
        dem: torch.Tensor = None,
        fast: bool = True,
        steps: int = 20,
    ) -> torch.Tensor:
        """Sample clean optical reconstruction from cloudy input liss4 conditioned on SAR & DEM.

        Args:
            liss4: Cloudy optical image (B, C, H, W)
            sar: Sentinel-1 SAR image (B, C_sar, H, W)
            dem: Optional DEM elevation features
            fast: If True, uses fast ODE sampling (20 steps); otherwise reverse SDE/DDPM
            steps: Number of integration steps
        """
        if self.use_ir_sde:
            if fast:
                return self.ir_sde.sample_ode(
                    self.unet,
                    y=liss4,
                    sar=sar,
                    steps=min(steps, self.noise_steps),
                    dem=dem,
                )
            else:
                return self.ir_sde.sample_sde(
                    self.unet,
                    y=liss4,
                    sar=sar,
                    steps=min(steps, self.noise_steps),
                    dem=dem,
                )

        b, c, h, w = liss4.shape
        device = liss4.device
        x_t = torch.randn(b, self.unet.outc.out_channels, h, w, device=device)

        for i in reversed(range(self.noise_steps)):
            t = torch.full((b,), i, device=device, dtype=torch.long)
            t_float = t.float() / self.noise_steps
            noise_pred = self.unet(liss4, sar, t_float, dem=dem)

            alpha = self.alphas[t].view(-1, 1, 1, 1)
            alpha_bar = self.alpha_bars[t].view(-1, 1, 1, 1)
            beta = self.betas[t].view(-1, 1, 1, 1)

            x_0_pred = (x_t - beta * noise_pred / (1 - alpha_bar).sqrt()) / alpha.sqrt()

            if i > 0:
                noise = torch.randn_like(x_t)
                alpha_bar_prev = self.alpha_bars[t - 1].view(-1, 1, 1, 1)
                posterior_var = beta * (1 - alpha_bar_prev / alpha_bar) / (1 - alpha_bar)
                x_t = x_0_pred + posterior_var.sqrt() * noise
            else:
                x_t = x_0_pred

        return x_t

    def q_sample(
        self,
        x_0: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor = None,
        y: torch.Tensor = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward diffusion perturbation."""
        if self.use_ir_sde and y is not None:
            t_norm = t.float() / self.noise_steps if t.dtype in (torch.int64, torch.int32) else t
            return self.ir_sde.q_sample(x_0, y, t_norm, noise=noise)

        if noise is None:
            noise = torch.randn_like(x_0)
        sqrt_ab = self.alpha_bars[t].sqrt().view(-1, 1, 1, 1)
        sqrt_1m_ab = (1 - self.alpha_bars[t]).sqrt().view(-1, 1, 1, 1)
        return sqrt_ab * x_0 + sqrt_1m_ab * noise, noise

    def compute_loss(
        self,
        clear: torch.Tensor,
        liss4: torch.Tensor,
        sar: torch.Tensor,
        dem: torch.Tensor = None,
        cloud_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """Compute training loss for SAR Conditional Diffusion (IR-SDE)."""
        b = clear.shape[0]
        device = clear.device

        if self.use_ir_sde:
            # Continuous normalized time t in [0, 1]
            t = torch.rand(b, device=device)
            x_t, noise = self.ir_sde.q_sample(x_0=clear, y=liss4, t=t)
            pred_x0 = self.unet(x_t, sar, t, dem=dem)

            # Reconstruction loss on target x_0
            rec_loss = F.l1_loss(pred_x0, clear)

            # Density/cloud weighted loss if mask provided
            if cloud_mask is not None:
                mask = cloud_mask.float()
                if mask.ndim == 3:
                    mask = mask.unsqueeze(1)
                if mask.shape[1] != clear.shape[1]:
                    mask = mask.expand(-1, clear.shape[1], -1, -1)
                dense_l1 = (torch.abs(pred_x0 - clear) * mask).sum() / (mask.sum() + 1e-7)
                rec_loss = rec_loss + 0.5 * dense_l1

            return rec_loss

        t = torch.randint(0, self.noise_steps, (b,), device=device)
        noise = torch.randn_like(clear)
        x_t, _ = self.q_sample(clear, t, noise)
        t_float = t.float() / self.noise_steps
        noise_pred = self.unet(liss4, sar, t_float, dem=dem)
        return F.mse_loss(noise_pred, noise)

