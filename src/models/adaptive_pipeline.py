import torch
import torch.nn as nn


class AdaptiveCloudRemoval(nn.Module):
    def __init__(self, density_net: nn.Module, correction_net: nn.Module,
                 temporal_fusion: nn.Module, sar_fusion: nn.Module,
                 thin_threshold: float = 0.3,
                 medium_threshold: float = 0.5,
                 dense_threshold: float = 0.8):
        super().__init__()
        self.density_net = density_net
        self.correction_net = correction_net
        self.temporal_fusion = temporal_fusion
        self.sar_fusion = sar_fusion
        self.thin_threshold = thin_threshold
        self.medium_threshold = medium_threshold
        self.dense_threshold = dense_threshold

    def forward(self, liss4: torch.Tensor, sar: torch.Tensor = None,
                temporal_refs: list[torch.Tensor] = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        density = self.density_net(liss4)

        thin_out = self.correction_net(liss4, density)
        medium_out = thin_out
        if temporal_refs and len(temporal_refs) > 0:
            medium_out = self.temporal_fusion(liss4, temporal_refs[0], density)

        dense_out = medium_out
        if sar is not None:
            dense_out = self.sar_fusion(liss4, sar)

        weights = self._compute_blend_weights(density)
        output = (
            weights[:, 0:1] * thin_out +
            weights[:, 1:2] * medium_out +
            weights[:, 2:3] * dense_out
        )

        return output, density, weights

    def _compute_blend_weights(self, density: torch.Tensor) -> torch.Tensor:
        b, _, h, w = density.shape
        weights = torch.zeros(b, 3, h, w, device=density.device)

        thin = (density[:, 0] < self.thin_threshold).float()
        medium = ((density[:, 0] >= self.thin_threshold) &
                  (density[:, 0] < self.dense_threshold)).float()
        dense = (density[:, 0] >= self.dense_threshold).float()

        total = thin + medium + dense + 1e-8
        weights[:, 0] = thin / total
        weights[:, 1] = medium / total
        weights[:, 2] = dense / total
        return weights


def build_adaptive_pipeline(thin_threshold: float = 0.3,
                            medium_threshold: float = 0.5,
                            dense_threshold: float = 0.8) -> AdaptiveCloudRemoval:
    from src.models.cloud_density import CloudDensityNet
    from src.models.thin_cloud_correction import ThinCloudCorrection
    from src.models.temporal_fusion import TemporalFusion
    from src.models.sar_fusion import SARConditionalUNet

    density_net = CloudDensityNet(in_channels=3, out_channels=1)
    correction_net = ThinCloudCorrection(in_channels=3)
    temporal_fusion = TemporalFusion(in_channels=3)
    sar_fusion = SARConditionalUNet(sar_channels=2, liss4_channels=3, out_channels=3)

    return AdaptiveCloudRemoval(
        density_net=density_net,
        correction_net=correction_net,
        temporal_fusion=temporal_fusion,
        sar_fusion=sar_fusion,
        thin_threshold=thin_threshold,
        medium_threshold=medium_threshold,
        dense_threshold=dense_threshold,
    )
