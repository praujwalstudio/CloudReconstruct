import torch
import torch.nn as nn
import torch.nn.functional as F


class SSIMLoss(nn.Module):
    def __init__(self, window_size: int = 11, sigma: float = 1.5):
        super().__init__()
        self.window_size = window_size
        self.sigma = sigma
        pad = window_size // 2
        self.pad = pad
        self.register_buffer("window", self._create_window(window_size, sigma))

    def _create_window(self, window_size: int, sigma: float) -> torch.Tensor:
        gauss = torch.arange(window_size, dtype=torch.float32) - window_size // 2
        gauss = torch.exp(-(gauss ** 2) / (2 * sigma ** 2))
        gauss = gauss / gauss.sum()
        window = gauss[:, None] * gauss[None, :]
        window = window.expand(1, 1, window_size, window_size)
        return window.contiguous()

    def forward(self, pred: torch.Tensor, target: torch.Tensor,
                data_range: float = 1.0) -> torch.Tensor:
        if pred.shape != target.shape:
            raise ValueError(f"Shape mismatch: {pred.shape} vs {target.shape}")

        c1 = (0.01 * data_range) ** 2
        c2 = (0.03 * data_range) ** 2

        mu1 = F.conv2d(pred, self.window, padding=self.pad, groups=pred.shape[1])
        mu2 = F.conv2d(target, self.window, padding=self.pad, groups=target.shape[1])
        mu1_sq, mu2_sq, mu12 = mu1 ** 2, mu2 ** 2, mu1 * mu2

        sigma1_sq = F.conv2d(pred ** 2, self.window, padding=self.pad, groups=pred.shape[1]) - mu1_sq
        sigma2_sq = F.conv2d(target ** 2, self.window, padding=self.pad, groups=target.shape[1]) - mu2_sq
        sigma12 = F.conv2d(pred * target, self.window, padding=self.pad, groups=pred.shape[1]) - mu12

        ssim_map = ((2 * mu12 + c1) * (2 * sigma12 + c2)) / \
                   ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
        return 1.0 - ssim_map.mean()


class SpectralAngleLoss(nn.Module):
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        dot = (pred * target).sum(dim=1)
        norm_p = torch.norm(pred, dim=1)
        norm_t = torch.norm(target, dim=1)
        cos_angle = dot / (norm_p * norm_t + self.eps)
        cos_angle = torch.clamp(cos_angle, -1 + self.eps, 1 - self.eps)
        angle = torch.acos(cos_angle)
        return angle.mean()


class PerceptualLoss(nn.Module):
    def __init__(self, device: str = "cpu"):
        super().__init__()
        from torchvision.models import vgg16, VGG16_Weights
        vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
        self.features = nn.Sequential(*list(vgg.features[:16])).to(device).eval()
        for p in self.features.parameters():
            p.requires_grad = False
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_norm = (pred - self.mean) / self.std
        target_norm = (target - self.mean) / self.std
        if pred_norm.shape[1] == 1:
            pred_norm = pred_norm.repeat(1, 3, 1, 1)
            target_norm = target_norm.repeat(1, 3, 1, 1)
        pred_feat = self.features(pred_norm)
        target_feat = self.features(target_norm)
        return F.l1_loss(pred_feat, target_feat)


class CombinedLoss(nn.Module):
    def __init__(self, weights: dict = None, device: str = "cpu"):
        super().__init__()
        self.weights = weights or {
            "l1": 1.0,
            "ssim": 1.0,
            "spectral": 0.5,
            "perceptual": 0.1,
        }
        self.l1 = nn.L1Loss()
        self.ssim = SSIMLoss()
        self.spectral = SpectralAngleLoss()
        self.perceptual = PerceptualLoss(device) if self.weights.get("perceptual", 0) > 0 else None

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> dict:
        losses = {}
        losses["l1"] = self.l1(pred, target)
        losses["ssim"] = self.ssim(pred, target)
        losses["spectral"] = self.spectral(pred, target)
        if self.perceptual is not None:
            losses["perceptual"] = self.perceptual(pred, target)
        total = sum(self.weights.get(k, 0) * v for k, v in losses.items())
        losses["total"] = total
        return losses
