from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage
from skimage.feature import peak_local_max
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from .common import clamp, natural_key, read_image
from .data import mask_to_points, read_mask
from .metrics import compute_count_metrics
from .patchwise import (
    _enable_mc_dropout,
    iter_patch_boxes,
    magnification_scale,
    pad_to_minimum,
    resize_grayscale,
    split_training_records,
    standardize_patch_tensor,
)


def build_center_heatmap(
    points: np.ndarray,
    shape: tuple[int, int],
    sigma: float = 2.0,
) -> np.ndarray:
    heatmap = np.zeros(shape, dtype=np.float32)
    if points.size == 0:
        return heatmap

    radius = int(max(2, round(3 * sigma)))
    yy, xx = np.mgrid[-radius : radius + 1, -radius : radius + 1]
    kernel = np.exp(-((xx ** 2 + yy ** 2) / (2.0 * sigma ** 2))).astype(np.float32)
    kernel /= float(kernel.max())

    height, width = shape
    for y, x in np.asarray(points, dtype=np.float32):
        iy = int(round(float(y)))
        ix = int(round(float(x)))
        y0 = max(0, iy - radius)
        y1 = min(height, iy + radius + 1)
        x0 = max(0, ix - radius)
        x1 = min(width, ix + radius + 1)

        ky0 = y0 - (iy - radius)
        ky1 = ky0 + (y1 - y0)
        kx0 = x0 - (ix - radius)
        kx1 = kx0 + (x1 - x0)
        heatmap[y0:y1, x0:x1] = np.maximum(heatmap[y0:y1, x0:x1], kernel[ky0:ky1, kx0:kx1])
    return heatmap


def build_rescaled_center_heatmap(mask: np.ndarray, scale: float, sigma: float = 2.0) -> np.ndarray:
    labels = np.asarray(mask)
    points = mask_to_points(labels)
    if points.size > 0 and abs(scale - 1.0) >= 1e-6:
        points = points * float(scale)
    target_shape = (
        max(16, int(round(labels.shape[0] * scale))),
        max(16, int(round(labels.shape[1] * scale))),
    )
    return build_center_heatmap(points, shape=target_shape, sigma=sigma)


def inverse_sigmoid(probability: float, eps: float = 1e-6) -> float:
    safe = min(max(float(probability), eps), 1.0 - eps)
    return math.log(safe / (1.0 - safe))


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GELU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)


def _resize_like(source: Tensor, reference: Tensor) -> Tensor:
    if source.shape[-2:] == reference.shape[-2:]:
        return source
    return F.interpolate(source, size=reference.shape[-2:], mode="bilinear", align_corners=False)


class CenterHeatmapUNet(nn.Module):
    def __init__(self, base_channels: int = 24, dropout: float = 0.10) -> None:
        super().__init__()
        self.enc1 = ConvBlock(1, base_channels, dropout=0.0)
        self.enc2 = ConvBlock(base_channels, base_channels * 2, dropout=dropout)
        self.bottleneck = ConvBlock(base_channels * 2, base_channels * 4, dropout=dropout)
        self.dec2 = ConvBlock(base_channels * 4 + base_channels * 2, base_channels * 2, dropout=dropout)
        self.dec1 = ConvBlock(base_channels * 2 + base_channels, base_channels, dropout=dropout)
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.head = nn.Conv2d(base_channels, 1, kernel_size=1)
        nn.init.zeros_(self.head.weight)
        nn.init.constant_(self.head.bias, -4.5)

    def forward(self, x: Tensor) -> Tensor:
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        bottleneck = self.bottleneck(self.pool(enc2))

        up2 = _resize_like(F.interpolate(bottleneck, scale_factor=2, mode="bilinear", align_corners=False), enc2)
        dec2 = self.dec2(torch.cat([up2, enc2], dim=1))

        up1 = _resize_like(F.interpolate(dec2, scale_factor=2, mode="bilinear", align_corners=False), enc1)
        dec1 = self.dec1(torch.cat([up1, enc1], dim=1))
        return self.head(dec1)


def build_center_model_from_checkpoint(checkpoint: dict[str, object]) -> CenterHeatmapUNet:
    model_kwargs = checkpoint.get("model_kwargs", {})
    model = CenterHeatmapUNet(**model_kwargs)
    model.load_state_dict(checkpoint["state_dict"])
    return model


def load_center_checkpoint(path: str | Path, device: str | torch.device = "cpu") -> tuple[CenterHeatmapUNet, dict[str, object]]:
    checkpoint = torch.load(Path(path), map_location=device)
    model = build_center_model_from_checkpoint(checkpoint)
    model.to(device)
    return model, checkpoint


class PatchCenterHeatmapDataset(Dataset):
    def __init__(
        self,
        records: pd.DataFrame,
        patch_size: int = 192,
        stride: int = 160,
        sigma: float = 2.0,
        reference_magnification: float = 40.0,
        min_scale: float = 0.50,
        max_scale: float = 4.00,
        empty_patch_keep_prob: float = 0.15,
        augment: bool = False,
        seed: int = 7,
        max_patches: int = 0,
    ) -> None:
        self.patch_size = patch_size
        self.augment = augment
        self.samples: list[tuple[np.ndarray, np.ndarray]] = []

        rng = np.random.default_rng(seed)
        source = records.reset_index(drop=True)
        for _, row in source.iterrows():
            if "mask_path" not in row or not isinstance(row["mask_path"], str) or not row["mask_path"]:
                raise ValueError("PatchCenterHeatmapDataset requires mask_path for every record.")

            image = read_image(row["image_path"])
            mask = read_mask(row["mask_path"])
            scale = magnification_scale(
                row.get("magnification"),
                reference_magnification=reference_magnification,
                min_scale=min_scale,
                max_scale=max_scale,
            )
            resized_image = pad_to_minimum(resize_grayscale(image, scale=scale, order=1), patch_size=patch_size)
            heatmap = pad_to_minimum(build_rescaled_center_heatmap(mask, scale=scale, sigma=sigma), patch_size=patch_size)

            for y0, x0 in iter_patch_boxes(resized_image.shape, patch_size=patch_size, stride=stride):
                image_patch = resized_image[y0 : y0 + patch_size, x0 : x0 + patch_size]
                heatmap_patch = heatmap[y0 : y0 + patch_size, x0 : x0 + patch_size]
                if float(heatmap_patch.max()) <= 1e-6 and rng.random() > empty_patch_keep_prob:
                    continue
                self.samples.append((image_patch.astype(np.float32, copy=False), heatmap_patch.astype(np.float32, copy=False)))
                if max_patches > 0 and len(self.samples) >= max_patches:
                    break
            if max_patches > 0 and len(self.samples) >= max_patches:
                break

        if not self.samples:
            raise ValueError("PatchCenterHeatmapDataset produced zero samples.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        image, heatmap = self.samples[index]
        image_patch = image.copy()
        heatmap_patch = heatmap.copy()
        if self.augment:
            if np.random.rand() < 0.5:
                image_patch = np.fliplr(image_patch).copy()
                heatmap_patch = np.fliplr(heatmap_patch).copy()
            if np.random.rand() < 0.5:
                image_patch = np.flipud(image_patch).copy()
                heatmap_patch = np.flipud(heatmap_patch).copy()
        return torch.from_numpy(image_patch[None, ...]), torch.from_numpy(heatmap_patch[None, ...])


def estimate_initial_center_bias(dataset: PatchCenterHeatmapDataset, eps: float = 1e-6) -> float:
    means = [float(sample[1].mean()) for sample in dataset.samples]
    return inverse_sigmoid(float(np.clip(np.mean(means), eps, 1.0 - eps)))


def sigmoid_focal_loss(logits: Tensor, targets: Tensor, alpha: float = 0.75, gamma: float = 2.0) -> Tensor:
    probabilities = torch.sigmoid(logits)
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    pt = probabilities * targets + (1.0 - probabilities) * (1.0 - targets)
    alpha_factor = alpha * targets + (1.0 - alpha) * (1.0 - targets)
    modulating_factor = (1.0 - pt).pow(gamma)
    return (alpha_factor * modulating_factor * bce).mean()


def soft_dice_loss(logits: Tensor, targets: Tensor, smooth: float = 1.0) -> Tensor:
    probabilities = torch.sigmoid(logits)
    intersection = (probabilities * targets).sum(dim=(-2, -1))
    union = probabilities.sum(dim=(-2, -1)) + targets.sum(dim=(-2, -1))
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return 1.0 - dice.mean()


def count_heatmap_peaks(
    heatmap: np.ndarray,
    threshold_abs: float = 0.35,
    min_distance: int = 3,
    smooth_sigma: float = 0.8,
) -> np.ndarray:
    field = np.asarray(heatmap, dtype=np.float32)
    if field.size == 0 or float(field.max()) <= 0.0:
        return np.empty((0, 2), dtype=np.int32)
    smooth = ndimage.gaussian_filter(field, sigma=smooth_sigma)
    coords = peak_local_max(
        smooth,
        min_distance=max(int(min_distance), 1),
        threshold_abs=float(threshold_abs),
        exclude_border=False,
    )
    return np.asarray(coords, dtype=np.int32)


def summarize_peak_grid(
    prediction_maps: list[np.ndarray],
    true_counts: list[float],
    thresholds: list[float],
    min_distances: list[int],
) -> dict[str, object]:
    best: dict[str, object] | None = None
    for threshold in thresholds:
        for min_distance in min_distances:
            counts = [
                float(len(count_heatmap_peaks(prediction_map, threshold_abs=threshold, min_distance=min_distance)))
                for prediction_map in prediction_maps
            ]
            metrics = compute_count_metrics(true_counts, counts)
            candidate = {
                "peak_threshold": float(threshold),
                "peak_min_distance": int(min_distance),
                "metrics": metrics,
            }
            if best is None or candidate["metrics"]["mape"] < best["metrics"]["mape"]:
                best = candidate
    if best is None:
        raise ValueError("Could not summarize peak grid without candidate parameters.")
    return best


def _predict_center_patch_batches(
    model: CenterHeatmapUNet,
    image: np.ndarray,
    device: torch.device,
    patch_size: int,
    stride: int,
    mc_samples: int,
    batch_size: int,
    input_normalization: str,
    peak_threshold: float,
    peak_min_distance: int,
) -> tuple[np.ndarray, float, float]:
    padded = pad_to_minimum(image, patch_size=patch_size)
    original_height, original_width = image.shape
    patch_boxes = iter_patch_boxes(padded.shape, patch_size=patch_size, stride=stride)
    patches = np.stack([padded[y0 : y0 + patch_size, x0 : x0 + patch_size] for y0, x0 in patch_boxes], axis=0)

    heatmap_accumulator = np.zeros_like(padded, dtype=np.float32)
    sample_counts: list[float] = []

    for sample_index in range(max(mc_samples, 1)):
        if sample_index > 0:
            _enable_mc_dropout(model)
        else:
            model.eval()

        sample_heatmap = np.zeros_like(padded, dtype=np.float32)
        sample_weight = np.zeros_like(padded, dtype=np.float32)

        with torch.no_grad():
            for start in range(0, len(patches), batch_size):
                batch_array = patches[start : start + batch_size]
                batch_tensor = torch.from_numpy(batch_array[:, None, ...]).to(device=device, dtype=torch.float32)
                if input_normalization == "zscore":
                    batch_tensor = standardize_patch_tensor(batch_tensor)
                predictions = torch.sigmoid(model(batch_tensor)).cpu().numpy()[:, 0]
                for prediction, (y0, x0) in zip(predictions, patch_boxes[start : start + batch_size]):
                    sample_heatmap[y0 : y0 + patch_size, x0 : x0 + patch_size] += prediction
                    sample_weight[y0 : y0 + patch_size, x0 : x0 + patch_size] += 1.0

        sample_heatmap /= np.maximum(sample_weight, 1.0)
        sample_heatmap = sample_heatmap[:original_height, :original_width]
        heatmap_accumulator[:original_height, :original_width] += sample_heatmap
        sample_counts.append(
            float(len(count_heatmap_peaks(sample_heatmap, threshold_abs=peak_threshold, min_distance=peak_min_distance)))
        )

    mean_heatmap = heatmap_accumulator[:original_height, :original_width] / float(max(mc_samples, 1))
    return mean_heatmap, float(np.mean(sample_counts)), float(np.std(sample_counts, ddof=0))


def predict_patchwise_center_heatmap_map(
    model: CenterHeatmapUNet,
    image: np.ndarray,
    device: torch.device,
    magnification: object | None = None,
    reference_magnification: float = 40.0,
    min_scale: float = 0.50,
    max_scale: float = 4.00,
    patch_size: int = 192,
    stride: int = 160,
    mc_samples: int = 1,
    batch_size: int = 4,
    input_normalization: str = "zscore",
    peak_threshold: float = 0.35,
    peak_min_distance: int = 3,
) -> dict[str, object]:
    scale_factor = magnification_scale(
        magnification,
        reference_magnification=reference_magnification,
        min_scale=min_scale,
        max_scale=max_scale,
    )
    resized = resize_grayscale(image, scale=scale_factor, order=1)
    mean_heatmap, count_mean, count_std = _predict_center_patch_batches(
        model=model,
        image=resized,
        device=device,
        patch_size=patch_size,
        stride=stride,
        mc_samples=mc_samples,
        batch_size=batch_size,
        input_normalization=input_normalization,
        peak_threshold=peak_threshold,
        peak_min_distance=peak_min_distance,
    )
    return {
        "heatmap": mean_heatmap,
        "count": count_mean,
        "count_std": count_std,
        "scale_factor": scale_factor,
    }


def predict_patchwise_center_heatmap(
    model: CenterHeatmapUNet,
    image: np.ndarray,
    device: torch.device,
    magnification: object | None = None,
    reference_magnification: float = 40.0,
    min_scale: float = 0.50,
    max_scale: float = 4.00,
    patch_size: int = 192,
    stride: int = 160,
    mc_samples: int = 1,
    batch_size: int = 4,
    input_normalization: str = "zscore",
    peak_threshold: float = 0.35,
    peak_min_distance: int = 3,
) -> dict[str, object]:
    result = predict_patchwise_center_heatmap_map(
        model=model,
        image=image,
        device=device,
        magnification=magnification,
        reference_magnification=reference_magnification,
        min_scale=min_scale,
        max_scale=max_scale,
        patch_size=patch_size,
        stride=stride,
        mc_samples=mc_samples,
        batch_size=batch_size,
        input_normalization=input_normalization,
        peak_threshold=peak_threshold,
        peak_min_distance=peak_min_distance,
    )
    peak_points = count_heatmap_peaks(
        np.asarray(result["heatmap"]),
        threshold_abs=peak_threshold,
        min_distance=peak_min_distance,
    )
    scale_factor = float(result["scale_factor"])
    if peak_points.size > 0 and abs(scale_factor - 1.0) >= 1e-6:
        peak_points = peak_points / scale_factor
    result["peak_points"] = peak_points
    result["count"] = float(len(peak_points))
    return result
