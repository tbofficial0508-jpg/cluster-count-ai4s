from __future__ import annotations

import re
from pathlib import Path
import math

import numpy as np
import pandas as pd
from skimage.transform import resize
import torch
from torch import Tensor
from torch.utils.data import Dataset

from .common import clamp, natural_key, read_image
from .data import gaussian_density_from_points, mask_to_points, read_mask
from .modeling import DensityCountingCNN
from .visualization import find_density_peaks


MAGNIFICATION_PATTERN = re.compile(r"([0-9]+(?:\.[0-9]+)?)\s*x", flags=re.IGNORECASE)


def parse_magnification(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, float) and np.isnan(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    match = MAGNIFICATION_PATTERN.search(text)
    if match:
        return float(match.group(1))
    numeric = re.search(r"([0-9]+(?:\.[0-9]+)?)", text)
    if numeric:
        return float(numeric.group(1))
    return None


def magnification_scale(
    value: object,
    reference_magnification: float = 40.0,
    min_scale: float = 0.50,
    max_scale: float = 4.00,
) -> float:
    magnification = parse_magnification(value)
    if magnification is None or magnification <= 0.0 or reference_magnification <= 0.0:
        return 1.0
    scale = float(reference_magnification) / float(magnification)
    return clamp(scale, low=min_scale, high=max_scale)


def resize_grayscale(array: np.ndarray, scale: float, order: int = 1) -> np.ndarray:
    image = np.asarray(array, dtype=np.float32)
    if abs(scale - 1.0) < 1e-6:
        return image
    target_height = max(16, int(round(image.shape[0] * scale)))
    target_width = max(16, int(round(image.shape[1] * scale)))
    resized = resize(
        image,
        output_shape=(target_height, target_width),
        order=order,
        mode="reflect",
        preserve_range=True,
        anti_aliasing=bool(order > 0 and scale < 1.0),
    )
    return np.asarray(resized, dtype=np.float32)


def build_rescaled_density(mask: np.ndarray, scale: float, sigma: float = 2.0) -> np.ndarray:
    labels = np.asarray(mask)
    points = mask_to_points(labels)
    if points.size > 0 and abs(scale - 1.0) >= 1e-6:
        points = points * float(scale)
    target_shape = (
        max(16, int(round(labels.shape[0] * scale))),
        max(16, int(round(labels.shape[1] * scale))),
    )
    return gaussian_density_from_points(points, shape=target_shape, sigma=sigma)


def pad_to_minimum(array: np.ndarray, patch_size: int) -> np.ndarray:
    image = np.asarray(array, dtype=np.float32)
    height, width = image.shape[:2]
    pad_height = max(0, patch_size - height)
    pad_width = max(0, patch_size - width)
    if pad_height == 0 and pad_width == 0:
        return image
    return np.pad(image, ((0, pad_height), (0, pad_width)), mode="constant")


def compute_patch_starts(length: int, patch_size: int, stride: int) -> list[int]:
    if length <= patch_size:
        return [0]
    starts = list(range(0, max(length - patch_size + 1, 1), stride))
    last_start = length - patch_size
    if not starts or starts[-1] != last_start:
        starts.append(last_start)
    return starts


def iter_patch_boxes(shape: tuple[int, int], patch_size: int, stride: int) -> list[tuple[int, int]]:
    height, width = shape
    y_starts = compute_patch_starts(height, patch_size, stride)
    x_starts = compute_patch_starts(width, patch_size, stride)
    return [(y0, x0) for y0 in y_starts for x0 in x_starts]


def _enable_mc_dropout(model: torch.nn.Module) -> None:
    model.eval()
    for module in model.modules():
        if isinstance(module, (torch.nn.Dropout, torch.nn.Dropout2d, torch.nn.Dropout3d)):
            module.train()


def standardize_patch_tensor(batch: Tensor, eps: float = 1e-4) -> Tensor:
    mean = batch.mean(dim=(-2, -1), keepdim=True)
    std = batch.std(dim=(-2, -1), keepdim=True).clamp_min(eps)
    return (batch - mean) / std


def inverse_softplus(value: float, eps: float = 1e-8) -> float:
    safe_value = max(float(value), eps)
    return math.log(math.expm1(safe_value))


class PatchDensityDataset(Dataset):
    def __init__(
        self,
        records: pd.DataFrame,
        patch_size: int = 256,
        stride: int = 224,
        sigma: float = 2.0,
        reference_magnification: float = 40.0,
        min_scale: float = 0.50,
        max_scale: float = 4.00,
        empty_patch_keep_prob: float = 0.20,
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
                raise ValueError("PatchDensityDataset requires mask_path for every record.")

            image = read_image(row["image_path"])
            mask = read_mask(row["mask_path"])
            scale = magnification_scale(
                row.get("magnification"),
                reference_magnification=reference_magnification,
                min_scale=min_scale,
                max_scale=max_scale,
            )
            resized_image = pad_to_minimum(resize_grayscale(image, scale=scale, order=1), patch_size=patch_size)
            density = pad_to_minimum(build_rescaled_density(mask, scale=scale, sigma=sigma), patch_size=patch_size)

            for y0, x0 in iter_patch_boxes(resized_image.shape, patch_size=patch_size, stride=stride):
                image_patch = resized_image[y0 : y0 + patch_size, x0 : x0 + patch_size]
                density_patch = density[y0 : y0 + patch_size, x0 : x0 + patch_size]
                if float(density_patch.sum()) <= 1e-6 and rng.random() > empty_patch_keep_prob:
                    continue
                self.samples.append((image_patch.astype(np.float32, copy=False), density_patch.astype(np.float32, copy=False)))
                if max_patches > 0 and len(self.samples) >= max_patches:
                    break
            if max_patches > 0 and len(self.samples) >= max_patches:
                break

        if not self.samples:
            raise ValueError("PatchDensityDataset produced zero samples.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        image, density = self.samples[index]
        image_patch = image.copy()
        density_patch = density.copy()
        if self.augment:
            if np.random.rand() < 0.5:
                image_patch = np.fliplr(image_patch).copy()
                density_patch = np.fliplr(density_patch).copy()
            if np.random.rand() < 0.5:
                image_patch = np.flipud(image_patch).copy()
                density_patch = np.flipud(density_patch).copy()
        return torch.from_numpy(image_patch[None, ...]), torch.from_numpy(density_patch[None, ...])


def split_training_records(
    records: pd.DataFrame,
    train_split: str = "train",
    val_split: str = "val",
    holdout_fraction: float = 0.20,
    seed: int = 7,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = records.copy()
    if val_split in set(frame["split"]):
        train_records = frame[frame["split"] == train_split].reset_index(drop=True)
        val_records = frame[frame["split"] == val_split].reset_index(drop=True)
        return train_records, val_records

    if train_split in set(frame["split"]):
        source = frame[frame["split"] == train_split].reset_index(drop=True)
    else:
        source = frame.reset_index(drop=True)

    if len(source) < 2:
        raise ValueError("Need at least two records to create a train/validation split.")

    source = source.sort_values("image_name", key=lambda series: series.map(natural_key)).reset_index(drop=True)
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(source))
    val_count = max(1, int(round(len(source) * holdout_fraction)))
    val_indices = set(indices[:val_count].tolist())
    train_records = source[[index not in val_indices for index in range(len(source))]].reset_index(drop=True)
    val_records = source[[index in val_indices for index in range(len(source))]].reset_index(drop=True)
    if len(train_records) == 0 or len(val_records) == 0:
        raise ValueError("Train/validation split resulted in an empty partition.")
    return train_records, val_records


def _predict_patch_batches(
    model: DensityCountingCNN,
    image: np.ndarray,
    device: torch.device,
    patch_size: int,
    stride: int,
    mc_samples: int,
    batch_size: int,
    input_normalization: str = "zscore",
) -> tuple[np.ndarray, float, float]:
    padded = pad_to_minimum(image, patch_size=patch_size)
    original_height, original_width = image.shape
    patch_boxes = iter_patch_boxes(padded.shape, patch_size=patch_size, stride=stride)
    patches = np.stack([padded[y0 : y0 + patch_size, x0 : x0 + patch_size] for y0, x0 in patch_boxes], axis=0)

    density_accumulator = np.zeros_like(padded, dtype=np.float32)
    sample_counts: list[float] = []

    for sample_index in range(max(mc_samples, 1)):
        if sample_index > 0:
            _enable_mc_dropout(model)
        else:
            model.eval()

        sample_density = np.zeros_like(padded, dtype=np.float32)
        sample_weight = np.zeros_like(padded, dtype=np.float32)

        with torch.no_grad():
            for start in range(0, len(patches), batch_size):
                batch_array = patches[start : start + batch_size]
                batch_tensor = torch.from_numpy(batch_array[:, None, ...]).to(device=device, dtype=torch.float32)
                if input_normalization == "zscore":
                    batch_tensor = standardize_patch_tensor(batch_tensor)
                predictions = model(batch_tensor).cpu().numpy()[:, 0]
                for prediction, (y0, x0) in zip(predictions, patch_boxes[start : start + batch_size]):
                    sample_density[y0 : y0 + patch_size, x0 : x0 + patch_size] += prediction
                    sample_weight[y0 : y0 + patch_size, x0 : x0 + patch_size] += 1.0

        sample_density /= np.maximum(sample_weight, 1.0)
        sample_density = sample_density[:original_height, :original_width]
        density_accumulator[:original_height, :original_width] += sample_density
        sample_counts.append(float(sample_density.sum()))

    mean_density = density_accumulator[:original_height, :original_width] / float(max(mc_samples, 1))
    return mean_density, float(np.mean(sample_counts)), float(np.std(sample_counts, ddof=0))


def predict_patchwise_density_map(
    model: DensityCountingCNN,
    image: np.ndarray,
    device: torch.device,
    magnification: object | None = None,
    reference_magnification: float = 40.0,
    min_scale: float = 0.50,
    max_scale: float = 4.00,
    patch_size: int = 256,
    stride: int = 224,
    mc_samples: int = 8,
    batch_size: int = 4,
    input_normalization: str = "zscore",
) -> dict[str, object]:
    scale_factor = magnification_scale(
        magnification,
        reference_magnification=reference_magnification,
        min_scale=min_scale,
        max_scale=max_scale,
    )
    resized = resize_grayscale(image, scale=scale_factor, order=1)
    mean_density, count_mean, count_std = _predict_patch_batches(
        model=model,
        image=resized,
        device=device,
        patch_size=patch_size,
        stride=stride,
        mc_samples=mc_samples,
        batch_size=batch_size,
        input_normalization=input_normalization,
    )
    peak_points = find_density_peaks(mean_density, expected_count=count_mean)
    if peak_points.size > 0 and abs(scale_factor - 1.0) >= 1e-6:
        peak_points = peak_points / float(scale_factor)
    return {
        "density_map": mean_density,
        "count": count_mean,
        "count_std": count_std,
        "peak_points": peak_points,
        "scale_factor": scale_factor,
    }
