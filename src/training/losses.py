from typing import Optional, Union, Tuple, Dict
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

        window = self.window.to(pred.device).expand(pred.shape[1], 1, self.window_size, self.window_size)
        mu1 = F.conv2d(pred, window, padding=self.pad, groups=pred.shape[1])
        mu2 = F.conv2d(target, window, padding=self.pad, groups=target.shape[1])
        mu1_sq, mu2_sq, mu12 = mu1 ** 2, mu2 ** 2, mu1 * mu2

        sigma1_sq = F.conv2d(pred ** 2, window, padding=self.pad, groups=pred.shape[1]) - mu1_sq
        sigma2_sq = F.conv2d(target ** 2, window, padding=self.pad, groups=target.shape[1]) - mu2_sq
        sigma12 = F.conv2d(pred * target, window, padding=self.pad, groups=pred.shape[1]) - mu12

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


class FilteredJaccardLoss(nn.Module):
    r"""Filtered Jaccard Loss (FJL) to prevent clear-sky over-penalization:
        FJL(t, y) = k_G * GL(t, y) * LP_{pc}(S) + k_J * JL(t, y) * HP_{p'_c}(S)

    Where:
        S = mean target cloud probability
        LP_{pc}(S) = 1 / (1 + exp(m * (S - pc)))
        HP_{p'_c}(S) = 1 / (1 + exp(m * (-S + p'_c)))
        m = 1000.0, pc = p'_c = 0.5, k_G = k_J = 1.0.
        GL can be 'GL1' (Inverted Jaccard) or 'GL2' (Normalized Binary Cross-Entropy).
    """

    def __init__(
        self,
        m: float = 1000.0,
        pc: float = 0.5,
        p_prime_c: float = 0.5,
        k_g: float = 1.0,
        k_j: float = 1.0,
        gl_type: str = "GL1",
        eps: float = 1e-7,
    ):
        super().__init__()
        self.m = float(m)
        self.pc = float(pc)
        self.p_prime_c = float(p_prime_c)
        self.k_g = float(k_g)
        self.k_j = float(k_j)
        self.gl_type = gl_type.upper()
        self.eps = eps

        if self.gl_type not in ("GL1", "GL2"):
            raise ValueError(f"gl_type must be 'GL1' or 'GL2', got {gl_type}")

    def _jaccard_loss(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Standard Soft Jaccard / IoU loss: 1 - intersection / union."""
        intersection = (t * y).sum(dim=(-2, -1))
        union = t.sum(dim=(-2, -1)) + y.sum(dim=(-2, -1)) - intersection
        iou = (intersection + self.eps) / (union + self.eps)
        return 1.0 - iou

    def _inverted_jaccard_loss(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """GL1: Inverted Jaccard loss on background (clear sky)."""
        t_inv = 1.0 - t
        y_inv = 1.0 - y
        intersection = (t_inv * y_inv).sum(dim=(-2, -1))
        union = t_inv.sum(dim=(-2, -1)) + y_inv.sum(dim=(-2, -1)) - intersection
        iou = (intersection + self.eps) / (union + self.eps)
        return 1.0 - iou

    def _normalized_bce(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """GL2: Normalized Binary Cross-Entropy."""
        y_clamped = torch.clamp(y, self.eps, 1.0 - self.eps)
        bce = -(t * torch.log(y_clamped) + (1.0 - t) * torch.log(1.0 - y_clamped))
        return bce.mean(dim=(-2, -1))

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Calculates Filtered Jaccard Loss.

        Args:
            pred (y): Predicted cloud probabilities in [0, 1].
            target (t): Ground truth cloud probabilities/binary mask in [0, 1].
        """
        if pred.shape != target.shape:
            raise ValueError(f"Shape mismatch in FilteredJaccardLoss: {pred.shape} vs {target.shape}")

        # S is mean target cloud probability across spatial dimensions
        S = target.mean(dim=(-2, -1), keepdim=True)

        # Continuous sigmoid transition filters with clamp for numerical stability
        lp_arg = torch.clamp(self.m * (S - self.pc), -50.0, 50.0)
        hp_arg = torch.clamp(self.m * (-S + self.p_prime_c), -50.0, 50.0)

        LP_pc = 1.0 / (1.0 + torch.exp(lp_arg))
        HP_p_prime_c = 1.0 / (1.0 + torch.exp(hp_arg))

        # Jaccard Loss (JL)
        JL = self._jaccard_loss(target, pred)

        # Global Loss (GL1 or GL2)
        if self.gl_type == "GL1":
            GL = self._inverted_jaccard_loss(target, pred)
        else:
            GL = self._normalized_bce(target, pred)

        # Broadcast filters if needed
        LP = LP_pc.squeeze(-1).squeeze(-1) if LP_pc.ndim > JL.ndim else LP_pc
        HP = HP_p_prime_c.squeeze(-1).squeeze(-1) if HP_p_prime_c.ndim > JL.ndim else HP_p_prime_c

        fjl = self.k_g * GL * LP + self.k_j * JL * HP
        return fjl.mean()


class CloudConstrainedLoss(nn.Module):
    r"""Cloud-Constrained Loss (CCL) to address temporal surface deviations:
        Lccl = lambda_ccl * L1_{cloudy} + L1_{clear}

    Calculates:
    - L1 distance inside cloud-and-shadow regions (where mask == 1).
    - L1 distance strictly in clear regions (where mask == 0).
    """

    def __init__(self, lambda_ccl: float = 1.0, eps: float = 1e-7):
        super().__init__()
        self.lambda_ccl = float(lambda_ccl)
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor,
                cloud_mask: torch.Tensor = None) -> torch.Tensor:
        """Calculates Cloud-Constrained Loss.

        Args:
            pred: Predicted image tensor (B, C, H, W).
            target: Clear ground truth target tensor (B, C, H, W).
            cloud_mask: Cloud and shadow mask tensor (B, 1, H, W) or (B, H, W) where 1=cloud/shadow, 0=clear.
        """
        if pred.shape != target.shape:
            raise ValueError(f"Shape mismatch in CloudConstrainedLoss: {pred.shape} vs {target.shape}")

        l1_diff = torch.abs(pred - target)

        if cloud_mask is None:
            return l1_diff.mean()

        if cloud_mask.ndim == 3:
            cloud_mask = cloud_mask.unsqueeze(1)

        # Ensure mask is float in [0, 1] and broadcastable to pred channels
        mask = cloud_mask.float()
        if mask.shape[1] != pred.shape[1]:
            mask = mask.expand(-1, pred.shape[1], -1, -1)

        clear_mask = 1.0 - mask

        cloudy_denom = mask.sum() + self.eps
        clear_denom = clear_mask.sum() + self.eps

        l1_cloudy = (l1_diff * mask).sum() / cloudy_denom
        l1_clear = (l1_diff * clear_mask).sum() / clear_denom

        return self.lambda_ccl * l1_cloudy + l1_clear


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
        self.fjl = FilteredJaccardLoss() if self.weights.get("fjl", 0) > 0 else None
        self.ccl = CloudConstrainedLoss(lambda_ccl=self.weights.get("lambda_ccl", 1.0)) if self.weights.get("ccl", 0) > 0 else None

    def forward(self, pred: torch.Tensor, target: torch.Tensor,
                cloud_mask: torch.Tensor = None) -> dict:
        losses = {}
        if self.weights.get("l1", 0) > 0:
            losses["l1"] = self.l1(pred, target)
        if self.weights.get("ssim", 0) > 0:
            losses["ssim"] = self.ssim(pred, target)
        if self.weights.get("spectral", 0) > 0:
            losses["spectral"] = self.spectral(pred, target)
        if self.perceptual is not None:
            losses["perceptual"] = self.perceptual(pred, target)
        if self.fjl is not None:
            losses["fjl"] = self.fjl(pred, target)
        if self.ccl is not None:
            losses["ccl"] = self.ccl(pred, target, cloud_mask=cloud_mask)

        total = sum(self.weights.get(k, 0) * v for k, v in losses.items())
        losses["total"] = total
        return losses


class MultiScaleConvergenceLoss(nn.Module):
    """
    Combines L1 Pixel-wise Loss, SSIM Loss, and SAM (Spectral Angle) Loss
    to accelerate deep model convergence and preserve spectral-structural details.
    Formula:
        loss = alpha * loss_ssim + (1.0 - alpha) * loss_l1 + lambda_sam * loss_sam
    """
    def __init__(self, alpha: float = 0.84, lambda_sam: float = 0.1):
        super().__init__()
        self.alpha = alpha
        self.lambda_sam = lambda_sam
        self.l1 = nn.L1Loss()
        self.ssim = SSIMLoss()
        self.sam = SpectralAngleLoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss_l1 = self.l1(pred, target)
        loss_ssim = self.ssim(pred, target)
        loss_sam = self.sam(pred, target)
        return self.alpha * loss_ssim + (1.0 - self.alpha) * loss_l1 + self.lambda_sam * loss_sam


class SpatialAttentionLoss(nn.Module):
    r"""Spatial Attention Loss (SpA-GAN):
        L_att = || A \odot (pred - gt) ||_1 / (|| A ||_1 + eps)
    Directs reconstruction penalty specifically to cloud/haze regions identified by attention map A.
    """

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        attention_map: torch.Tensor,
    ) -> torch.Tensor:
        if attention_map.ndim == 3:
            attention_map = attention_map.unsqueeze(1)
        if attention_map.shape[1] != pred.shape[1]:
            attention_map = attention_map.expand(-1, pred.shape[1], -1, -1)

        diff = torch.abs(pred - target)
        weighted_diff = diff * attention_map
        return weighted_diff.sum() / (attention_map.sum() + self.eps)


class LSGANLoss(nn.Module):
    """Least Squares GAN Loss (LSGAN) for stable adversarial training.

    Minimizes Pearson divergence for smoother gradients and higher visual fidelity.
    """

    def __init__(self, target_real: float = 1.0, target_fake: float = 0.0):
        super().__init__()
        self.target_real = target_real
        self.target_fake = target_fake

    def generator_loss(self, d_fake: torch.Tensor) -> torch.Tensor:
        """G wants D(fake) to be classified as real."""
        return 0.5 * torch.mean((d_fake - self.target_real) ** 2)

    def discriminator_loss(self, d_real: torch.Tensor, d_fake: torch.Tensor) -> torch.Tensor:
        """D wants D(real) -> real and D(fake) -> fake."""
        loss_real = 0.5 * torch.mean((d_real - self.target_real) ** 2)
        loss_fake = 0.5 * torch.mean((d_fake - self.target_fake) ** 2)
        return loss_real + loss_fake


class CARLLoss(nn.Module):
    r"""Cloud-Adaptive Reconstruction Loss (CARL / DSEN2-CR):
        L_carl = L_rec + lambda_adv * L_gan + lambda_att * L_att
    """

    def __init__(
        self,
        lambda_l1: float = 1.0,
        lambda_ssim: float = 0.5,
        lambda_att: float = 0.5,
        lambda_adv: float = 0.01,
        device: str = "cpu",
    ):
        super().__init__()
        self.lambda_l1 = lambda_l1
        self.lambda_ssim = lambda_ssim
        self.lambda_att = lambda_att
        self.lambda_adv = lambda_adv

        self.l1 = nn.L1Loss()
        self.ssim = SSIMLoss()
        self.att_loss = SpatialAttentionLoss()
        self.lsgan = LSGANLoss()

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        attention_map: Optional[torch.Tensor] = None,
        d_fake: Optional[torch.Tensor] = None,
    ) -> dict:
        losses = {}
        losses["l1"] = self.l1(pred, target)
        losses["ssim"] = self.ssim(pred, target)

        total = self.lambda_l1 * losses["l1"] + self.lambda_ssim * losses["ssim"]

        if attention_map is not None and self.lambda_att > 0:
            losses["att"] = self.att_loss(pred, target, attention_map)
            total = total + self.lambda_att * losses["att"]

        if d_fake is not None and self.lambda_adv > 0:
            losses["adv"] = self.lsgan.generator_loss(d_fake)
            total = total + self.lambda_adv * losses["adv"]

        losses["total"] = total
        return losses


class HeteroscedasticNLLLoss(nn.Module):
    r"""Heteroscedastic Gaussian Negative Log-Likelihood Loss (Kendall & Gal / UnCRtainTS):
        L_NLL(pred, target, log_var) = 0.5 * exp(-log_var) * ||target - pred||^2 + 0.5 * log_var
    Enables single-pass calibrated uncertainty estimation alongside surface reconstruction.
    """

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        log_var: torch.Tensor,
    ) -> torch.Tensor:
        if pred.shape != target.shape:
            raise ValueError(f"Shape mismatch: pred {pred.shape} vs target {target.shape}")
        if log_var.shape != pred.shape:
            log_var = log_var.expand_as(pred)

        # Stable precision weighting: exp(-s) * squared_diff + s
        precision = torch.exp(-log_var)
        diff_sq = (pred - target) ** 2
        loss = 0.5 * (precision * diff_sq + log_var)
        return loss.mean()



