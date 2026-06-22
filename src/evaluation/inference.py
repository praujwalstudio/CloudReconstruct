import math
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path

from src.config import DENSITY_CKPT, DIFFUSION_CKPT, CHECKPOINTS
from src.models.adaptive_pipeline import (
    build_adaptive_pipeline,
    AdaptiveCloudRemoval,
)
from src.evaluation.confidence import ConfidenceMap
from src.evaluation.analysis_readiness import AnalysisReadiness
from src.evaluation.geotiff_output import write_analysis_ready_product


def _load_checkpoint(path, map_location: str = "cpu") -> dict | None:
    if path is None:
        return None
    path = Path(path)
    if path.is_dir():
        files = sorted(path.glob("*.pt")) + sorted(path.glob("*.pth"))
        if files:
            path = files[0]
        else:
            return None
    if not path.exists():
        return None
    return torch.load(path, map_location=map_location, weights_only=False)


def _numpy_to_tensor(image: np.ndarray, device: str = "cpu",
                     data_max: float = None) -> torch.Tensor:
    if data_max is not None:
        image = image.astype(np.float32) / float(data_max)
    elif image.dtype == np.uint16:
        image = image.astype(np.float32) / 65535.0
    elif image.dtype == np.uint8:
        image = image.astype(np.float32) / 255.0
    else:
        image = image.astype(np.float32)
    tensor = torch.from_numpy(image)
    if tensor.ndim == 3:
        tensor = tensor.permute(2, 0, 1)
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0)
    if tensor.shape[0] == 1:
        tensor = tensor.repeat(3, 1, 1)
    return tensor.unsqueeze(0).to(device)


def _tensor_to_numpy(tensor: torch.Tensor, dtype=np.uint16) -> np.ndarray:
    arr = tensor.detach().cpu().squeeze(0).permute(1, 2, 0).numpy()
    arr = np.clip(arr, 0, 1)
    if dtype == np.uint16:
        arr = (arr * 65535).astype(np.uint16)
    elif dtype == np.uint8:
        arr = (arr * 255).astype(np.uint8)
    return arr


def _find_latest_checkpoint_dir(base_dir: Path, suffix: str = "") -> Path:
    candidates = sorted(base_dir.glob(f"*{suffix}"))
    if candidates:
        return candidates[-1]
    return base_dir


class CloudFreeInference:
    def __init__(self, device: str = "cpu",
                 density_ckpt: Path = None,
                 correction_ckpt: Path = None,
                 sar_ckpt: Path = None,
                 temporal_ckpt: Path = None,
                 thin_threshold: float = 0.3,
                 medium_threshold: float = 0.5,
                 dense_threshold: float = 0.8):
        self.device = device if torch.cuda.is_available() else "cpu"
        if device == "cuda" and not torch.cuda.is_available():
            self.device = "cpu"

        self.pipeline = self._build_pipeline(
            density_ckpt, correction_ckpt, sar_ckpt, temporal_ckpt,
            thin_threshold, medium_threshold, dense_threshold,
        )
        self.pipeline.eval()
        self.confidence = ConfidenceMap()
        self.readiness = AnalysisReadiness()

    def _build_pipeline(self, density_ckpt, correction_ckpt, sar_ckpt,
                        temporal_ckpt, thin_threshold, medium_threshold,
                        dense_threshold) -> AdaptiveCloudRemoval:
        pipeline = build_adaptive_pipeline(thin_threshold, medium_threshold, dense_threshold)
        pipeline.to(self.device)

        density_sd = _load_checkpoint(density_ckpt or DENSITY_CKPT, self.device)
        if density_sd is not None:
            pipeline.density_net.load_state_dict(density_sd, strict=False)

        correction_sd = _load_checkpoint(correction_ckpt or DIFFUSION_CKPT, self.device)
        if correction_sd is not None:
            pipeline.correction_net.load_state_dict(correction_sd, strict=False)

        sar_sd = _load_checkpoint(sar_ckpt, self.device)
        if sar_sd is not None:
            pipeline.sar_fusion.load_state_dict(sar_sd, strict=False)

        temporal_sd = _load_checkpoint(temporal_ckpt, self.device)
        if temporal_sd is not None:
            pipeline.temporal_fusion.load_state_dict(temporal_sd, strict=False)

        return pipeline

    @staticmethod
    def _need_tiling(tensor: torch.Tensor, max_pixels: int = 262144) -> bool:
        return tensor.shape[-2] * tensor.shape[-1] > max_pixels

    @staticmethod
    def _make_blend_weights(size: int, device: torch.device) -> torch.Tensor:
        center = size // 2
        dist = (torch.arange(size, device=device).float() - center).abs_().clamp(0, center)
        w_1d = 0.5 + 0.5 * (1.0 - dist / center)
        weights = w_1d[:, None] * w_1d[None, :]
        return weights.view(1, 1, size, size)

    def _extract_patches(self, tensor: torch.Tensor, patch_size: int = 256,
                         overlap: int = 32) -> tuple[list, list]:
        stride = patch_size - overlap
        B, C, H, W = tensor.shape
        patches, coords = [], []
        for y in range(0, H, stride):
            for x in range(0, W, stride):
                patch = tensor[:, :, y:y + patch_size, x:x + patch_size]
                if patch.shape[-2] < patch_size or patch.shape[-1] < patch_size:
                    ph = patch_size - patch.shape[-2]
                    pw = patch_size - patch.shape[-1]
                    patch = F.pad(patch, (0, pw, 0, ph), mode="replicate")
                patches.append(patch)
                coords.append((y, x))
        return patches, coords

    def _reconstruct(self, patches: list, coords: list, output_shape: tuple,
                     patch_size: int, overlap: int) -> torch.Tensor:
        B, C, H, W = output_shape
        device = patches[0].device
        accum = torch.zeros(B, C, H, W, device=device)
        weight = torch.zeros(B, 1, H, W, device=device)
        weights = self._make_blend_weights(patch_size, device)
        for patch, (y, x) in zip(patches, coords):
            if y >= H or x >= W:
                continue
            y_end = min(y + patch_size, H)
            x_end = min(x + patch_size, W)
            ph = y_end - y
            pw = x_end - x
            accum[:, :, y:y_end, x:x_end] += patch[:, :, :ph, :pw] * weights[:, :, :ph, :pw]
            weight[:, :, y:y_end, x:x_end] += weights[:, :, :ph, :pw]
        return accum / (weight + 1e-8)

    def _process_tiled(self, in_tensor: torch.Tensor, sar_tensor: torch.Tensor = None,
                       ref_tensors: list[torch.Tensor] = None,
                       patch_size: int = 256, overlap: int = 32) -> tuple[torch.Tensor, torch.Tensor]:
        in_patches, coords = self._extract_patches(in_tensor, patch_size, overlap)
        sar_patches = None
        if sar_tensor is not None:
            sar_patches, _ = self._extract_patches(sar_tensor, patch_size, overlap)
        ref_patch_lists = None
        if ref_tensors is not None:
            ref_patch_lists = []
            for ref in ref_tensors:
                p, _ = self._extract_patches(ref, patch_size, overlap)
                ref_patch_lists.append(p)

        corrected_patches, density_patches = [], []
        for i, in_patch in enumerate(in_patches):
            sar_p = sar_patches[i] if sar_patches else None
            ref_p = [rl[i] for rl in ref_patch_lists] if ref_patch_lists else None
            corr, dens, _ = self.pipeline(in_patch, sar_p, ref_p)
            corrected_patches.append(corr)
            density_patches.append(dens)

        B, C_in, H_in, W_in = in_tensor.shape
        corr = self._reconstruct(corrected_patches, coords, (B, corr.shape[1], H_in, W_in),
                                 patch_size, overlap)
        dens = self._reconstruct(density_patches, coords, (B, dens.shape[1], H_in, W_in),
                                 patch_size, overlap)
        return corr, dens

    @torch.no_grad()
    def correct(self, image: np.ndarray, sar: np.ndarray = None,
                temporal_refs: list[np.ndarray] = None,
                dem_processor=None, sun_zenith: float = 45,
                sun_azimuth: float = 180,
                data_max: float = None) -> dict:
        in_tensor = _numpy_to_tensor(image, self.device, data_max)

        sar_tensor = None
        if sar is not None:
            sar_tensor = _numpy_to_tensor(sar, self.device)
            if sar_tensor.shape[1] > 2:
                sar_tensor = sar_tensor[:, :2]

        ref_tensors = None
        if temporal_refs:
            ref_tensors = [_numpy_to_tensor(r, self.device) for r in temporal_refs]

        if self._need_tiling(in_tensor):
            corrected_tensor, density_tensor = self._process_tiled(
                in_tensor, sar_tensor, ref_tensors
            )
        else:
            corrected_tensor, density_tensor, _ = self.pipeline(
                in_tensor, sar_tensor, ref_tensors
            )

        corrected = _tensor_to_numpy(corrected_tensor, image.dtype)
        density = _tensor_to_numpy(density_tensor, np.float32).squeeze(-1)

        if dem_processor is not None:
            corrected = dem_processor.correct(
                corrected, np.radians(sun_zenith), np.radians(sun_azimuth)
            )

        conf = self.confidence.compute(density)
        ref_image = image
        if ref_image.ndim == 2:
            ref_image = np.stack([ref_image] * 3, axis=-1)
        elif ref_image.ndim == 3 and ref_image.shape[2] == 1:
            ref_image = np.repeat(ref_image, 3, axis=2)
        ars = self.readiness.evaluate(conf, corrected, ref_image)

        return {
            "corrected": corrected,
            "density": density,
            "confidence": conf,
            "ars": ars,
            "dtype": image.dtype,
        }

    def correct_and_save(self, output_path: Path, image: np.ndarray,
                         sar: np.ndarray = None,
                         temporal_refs: list[np.ndarray] = None,
                         dem_processor=None, profile: dict = None,
                         metadata: dict = None, sun_zenith: float = 45,
                         sun_azimuth: float = 180) -> Path:
        result = self.correct(image, sar, temporal_refs, dem_processor,
                              sun_zenith, sun_azimuth)
        return write_analysis_ready_product(
            output_path, result["corrected"], result["confidence"],
            result["ars"], profile, metadata,
        )
