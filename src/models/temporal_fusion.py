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
    with torch.no_grad():
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1.0, 1.0, h, device=image.device),
            torch.linspace(-1.0, 1.0, w, device=image.device),
            indexing="ij",
        )
        grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).expand(b, -1, -1, -1).contiguous()
        flow_up = flow.permute(0, 2, 3, 1)
        grid = grid + flow_up / torch.tensor([w, h], device=image.device).view(1, 1, 1, 2)
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


class TemporalFusion(nn.Module):
    def __init__(self, in_channels: int = 3, hidden: int = 64):
        super().__init__()
        self.alignment = AlignmentNet(in_channels * 2)
        self.blend = nn.Sequential(
            nn.Conv2d(in_channels * 2 + 1, hidden, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.ReLU(inplace=True),
            SelfAttention2d(hidden),
            nn.Conv2d(hidden, in_channels, 3, padding=1),
        )

    def forward(self, cloudy: torch.Tensor, reference: torch.Tensor,
                density: torch.Tensor) -> torch.Tensor:
        flow = self.alignment(cloudy, reference)
        ref_aligned = apply_flow(reference, flow)
        x = torch.cat([cloudy, ref_aligned, density], dim=1)
        return self.blend(x)


class MultiTemporalFusion(nn.Module):
    def __init__(self, in_channels: int = 3, hidden: int = 64):
        super().__init__()
        self.fusion = TemporalFusion(in_channels, hidden)

    def forward(self, cloudy: torch.Tensor, references: list[torch.Tensor],
                density: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        if not references:
            return cloudy, []

        fused_list = []
        for ref in references:
            fused = self.fusion(cloudy, ref, density)
            fused_list.append(fused)

        stacked = torch.stack(fused_list, dim=0)
        weights = torch.ones(stacked.shape[0], device=cloudy.device)
        weights = weights / weights.sum()
        output = torch.einsum("k,kbchw->bchw", weights, stacked)
        return output, fused_list
