"""Temporal Fusion Module for Cloud Removal (WP1/WP3)
===================================================
Provides dense optical reconstruction via historical cloud-free optical references.
Uses an AlignmentNet for sub-pixel optical flow warping, Cross-Temporal Attention (U-TAE / UnCRtainTS),
Self-Attention feature refinement, and a Heteroscedastic Uncertainty Head (log-variance log sigma^2):
    prior_blend = cloudy * (1 - density) + ref_aligned * density
    reconstructed = prior_blend + delta_learned
    log_var = uncertainty_head(fused_features)
"""

from typing import Tuple, List, Union, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class AlignmentNet(nn.Module):
    def __init__(self, in_channels: int = 6):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 2, 3, padding=1),
        )

    def forward(self, cloudy: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        x = torch.cat([cloudy, reference], dim=1)
        flow = self.conv(x)
        flow = torch.tanh(flow) * 5.0
        return flow


def apply_flow(image: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    b, c, h, w = image.shape
    grid_y, grid_x = torch.meshgrid(
        torch.linspace(-1.0, 1.0, h, device=image.device),
        torch.linspace(-1.0, 1.0, w, device=image.device),
        indexing="ij",
    )
    grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).expand(b, -1, -1, -1).contiguous()
    flow_up = flow.permute(0, 2, 3, 1)
    grid = grid + flow_up / torch.tensor([w, h], device=image.device, dtype=flow.dtype).view(1, 1, 1, 2)
    grid = torch.clamp(grid, -1.0, 1.0)
    return F.grid_sample(image, grid, mode="bilinear", padding_mode="border", align_corners=True)


class SelfAttention2d(nn.Module):
    def __init__(self, dim: int, num_heads: int = 4, max_tokens: int = 1024):
        super().__init__()
        self.norm = nn.BatchNorm2d(dim)
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.to_qkv = nn.Conv2d(dim, dim * 3, 1)
        self.to_out = nn.Conv2d(dim, dim, 1)
        self.max_tokens = max_tokens

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        n_tokens = H * W
        qkv = self.to_qkv(self.norm(x))

        if n_tokens > self.max_tokens:
            ratio = (n_tokens / self.max_tokens) ** 0.5
            pool_h = max(1, int(H / ratio))
            pool_w = max(1, int(W / ratio))
            qkv = nn.functional.interpolate(qkv, size=(pool_h, pool_w), mode="area")
            ph, pw = pool_h, pool_w
        else:
            ph, pw = H, W

        q, k, v = qkv.chunk(3, dim=1)
        q = q.view(B, self.num_heads, C // self.num_heads, -1)
        k = k.view(B, self.num_heads, C // self.num_heads, -1)
        v = v.view(B, self.num_heads, C // self.num_heads, -1)
        attn = torch.softmax(q.transpose(-2, -1) @ k * self.scale, dim=-1)
        out = (v @ attn.transpose(-2, -1)).view(B, C, ph, pw)
        if n_tokens > self.max_tokens:
            out = nn.functional.interpolate(out, size=(H, W), mode="bilinear", align_corners=False)
        return x + self.to_out(out)


class CrossTemporalAttention2d(nn.Module):
    """Learnable Cross-Temporal Attention Module (U-TAE / UnCRtainTS style).

    Query comes from cloudy target image features; Key and Value come from clear historical reference features.
    """

    def __init__(self, in_dim: int, num_heads: int = 4, max_tokens: int = 1024):
        super().__init__()
        self.norm_q = nn.BatchNorm2d(in_dim)
        self.norm_kv = nn.BatchNorm2d(in_dim)
        self.num_heads = num_heads
        self.scale = (in_dim // num_heads) ** -0.5
        self.to_q = nn.Conv2d(in_dim, in_dim, 1)
        self.to_k = nn.Conv2d(in_dim, in_dim, 1)
        self.to_v = nn.Conv2d(in_dim, in_dim, 1)
        self.to_out = nn.Conv2d(in_dim, in_dim, 1)
        self.max_tokens = max_tokens

    def forward(self, target_feats: torch.Tensor, ref_feats: torch.Tensor) -> torch.Tensor:
        B, C, H, W = target_feats.shape
        n_tokens = H * W

        q = self.to_q(self.norm_q(target_feats))
        k = self.to_k(self.norm_kv(ref_feats))
        v = self.to_v(self.norm_kv(ref_feats))

        if n_tokens > self.max_tokens:
            ratio = (n_tokens / self.max_tokens) ** 0.5
            ph = max(1, int(H / ratio))
            pw = max(1, int(W / ratio))
            q = F.interpolate(q, size=(ph, pw), mode="area")
            k = F.interpolate(k, size=(ph, pw), mode="area")
            v = F.interpolate(v, size=(ph, pw), mode="area")
        else:
            ph, pw = H, W

        q = q.view(B, self.num_heads, C // self.num_heads, -1)
        k = k.view(B, self.num_heads, C // self.num_heads, -1)
        v = v.view(B, self.num_heads, C // self.num_heads, -1)

        attn = torch.softmax(q.transpose(-2, -1) @ k * self.scale, dim=-1)
        out = (v @ attn.transpose(-2, -1)).view(B, C, ph, pw)

        if n_tokens > self.max_tokens:
            out = F.interpolate(out, size=(H, W), mode="bilinear", align_corners=False)

        return target_feats + self.to_out(out)


class UncertaintyHead(nn.Module):
    """Predicts heteroscedastic log-variance s = log sigma^2 per pixel."""

    def __init__(self, in_features: int, out_channels: int = 3):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_features, in_features // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_features // 2, out_channels, kernel_size=1),
        )

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        # Clamp log-variance to [-10.0, 5.0] for strict numerical stability
        log_var = self.head(feats)
        return torch.clamp(log_var, min=-10.0, max=5.0)


class TemporalFusion(nn.Module):
    """Density-gated residual temporal fusion with Cross-Temporal Attention and Uncertainty Head.

    Prior baseline:
        prior_blend = cloudy * (1.0 - density) + ref_aligned * density
        reconstructed = prior_blend + delta_learned
    """

    def __init__(self, in_channels: int = 3, hidden: int = 64):
        super().__init__()
        self.in_channels = in_channels
        self.alignment = AlignmentNet(in_channels * 2)

        # Cross-temporal feature embedding
        self.target_encoder = nn.Conv2d(in_channels + 1, hidden, kernel_size=3, padding=1)
        self.ref_encoder = nn.Conv2d(in_channels, hidden, kernel_size=3, padding=1)
        self.cross_attn = CrossTemporalAttention2d(hidden)
        self.self_attn = SelfAttention2d(hidden)

        # Feature processing backbone
        self.feat_conv = nn.Sequential(
            nn.Conv2d(hidden * 2 + (in_channels * 2 + 1), hidden, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.delta_head = nn.Conv2d(hidden, in_channels, kernel_size=3, padding=1)
        self.uncertainty_head = UncertaintyHead(hidden, out_channels=in_channels)

    def forward(
        self,
        cloudy: torch.Tensor,
        reference: torch.Tensor,
        density: torch.Tensor,
        return_uncertainty: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        flow = self.alignment(cloudy, reference)
        ref_aligned = apply_flow(reference, flow)

        # 1. Physical prior baseline (density-gated blend)
        prior_blend = cloudy * (1.0 - density) + ref_aligned * density

        # 2. Cross-temporal learned attention alignment
        t_feats = self.target_encoder(torch.cat([cloudy, density], dim=1))
        r_feats = self.ref_encoder(ref_aligned)
        aligned_feats = self.cross_attn(t_feats, r_feats)
        aligned_feats = self.self_attn(aligned_feats)

        # 3. Concatenate and compute residual refinement + uncertainty
        x = torch.cat([aligned_feats, t_feats, cloudy, ref_aligned, density], dim=1)
        fused = self.feat_conv(x)

        delta = torch.tanh(self.delta_head(fused)) * 0.1
        log_var = self.uncertainty_head(fused)

        # Ensure computational graph connects to uncertainty head even when not returned
        reconstructed = torch.clamp(prior_blend + delta + 0.0 * log_var.sum(), 0.0, 1.0)

        if return_uncertainty:
            return reconstructed, log_var

        return reconstructed


# Alias for explicitly named references
ResidualTemporalFusion = TemporalFusion


class MultiTemporalFusion(nn.Module):
    def __init__(self, in_channels: int = 3, hidden: int = 64):
        super().__init__()
        self.fusion = TemporalFusion(in_channels, hidden)

    def forward(
        self,
        cloudy: torch.Tensor,
        references: List[torch.Tensor],
        density: torch.Tensor,
        return_uncertainty: bool = False,
    ) -> Union[
        Tuple[torch.Tensor, List[torch.Tensor]],
        Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]
    ]:
        if not references:
            if return_uncertainty:
                dummy_log_var = torch.zeros_like(cloudy)
                return cloudy, dummy_log_var, []
            return cloudy, []

        fused_list = []
        log_vars = []

        for ref in references:
            if return_uncertainty:
                fused, lv = self.fusion(cloudy, ref, density, return_uncertainty=True)
                fused_list.append(fused)
                log_vars.append(lv)
            else:
                fused = self.fusion(cloudy, ref, density)
                fused_list.append(fused)

        stacked = torch.stack(fused_list, dim=0)

        if return_uncertainty:
            # Inverse-variance weighted temporal fusion
            stacked_log_var = torch.stack(log_vars, dim=0)
            inv_vars = torch.exp(-stacked_log_var)  # 1 / sigma^2
            weights = inv_vars / (inv_vars.sum(dim=0, keepdim=True) + 1e-7)

            output = (stacked * weights).sum(dim=0)
            composite_var = 1.0 / (inv_vars.sum(dim=0) + 1e-7)
            composite_log_var = torch.log(composite_var.clamp(min=1e-7, max=100.0))

            return output, composite_log_var, fused_list

        weights = torch.ones(stacked.shape[0], device=cloudy.device)
        weights = weights / weights.sum()
        output = torch.einsum("k,kbchw->bchw", weights, stacked)
        return output, fused_list
