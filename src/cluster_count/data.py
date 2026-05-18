from __future__ import annotations

import csv
import re
import warnings
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import ndimage
from torch.utils.data import Dataset
import torch

from .common import ensure_dir, list_images, natural_key, normalize_density_map, read_image


S_BSST265_MARKERS = ("image_description.csv", "rawimages", "groundtruth")


def parse_count_from_filename(filename: str) -> int:
    name = Path(filename).stem
    explicit_patterns = [
        r"count[_-]?(\d+)",
        r"(?:^|[_-])C(\d+)(?:[_-]|$)",
        r"(?:^|[_-])N(\d+)(?:[_-]|$)",
    ]
    for pattern in explicit_patterns:
        match = re.search(pattern, name, flags=re.IGNORECASE)
        if match:
            return int(match.group(1))
    trailing = re.search(r"(\d+)$", name)
    if trailing:
        return int(trailing.group(1))
    raise ValueError(f"Could not parse count from filename: {filename}")


def count_instances(mask: np.ndarray) -> int:
    labels = np.unique(mask)
    return int((labels > 0).sum())


def mask_to_points(mask: np.ndarray) -> np.ndarray:
    labels = [int(label) for label in np.unique(mask) if label > 0]
    if not labels:
        return np.empty((0, 2), dtype=np.float32)
    centers = ndimage.center_of_mass(mask > 0, labels=mask, index=labels)
    return np.asarray(centers, dtype=np.float32)


def gaussian_density_from_points(
    points: np.ndarray,
    shape: tuple[int, int],
    sigma: float = 2.0,
) -> np.ndarray:
    density = np.zeros(shape, dtype=np.float32)
    if points.size == 0:
        return density

    radius = int(max(2, round(3 * sigma)))
    yy, xx = np.mgrid[-radius : radius + 1, -radius : radius + 1]
    kernel = np.exp(-((xx ** 2 + yy ** 2) / (2.0 * sigma ** 2))).astype(np.float32)
    kernel /= float(kernel.sum())

    height, width = shape
    for y, x in points:
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
        density[y0:y1, x0:x1] += kernel[ky0:ky1, kx0:kx1]

    return normalize_density_map(density, float(len(points)))


def density_from_mask(mask: np.ndarray, sigma: float = 2.0) -> np.ndarray:
    return gaussian_density_from_points(mask_to_points(mask), shape=mask.shape, sigma=sigma)


def discover_s_bsst265_root(start: str | Path | None = None) -> Path | None:
    candidates: list[Path] = []
    if start is not None:
        base = Path(start).resolve()
        candidates.extend([base, base / "dataset", base / "S-BSST265", base / "S-BSST265" / "dataset"])

    here = Path(__file__).resolve()
    for parent in here.parents:
        candidates.extend([parent / "S-BSST265", parent / "S-BSST265" / "dataset"])

    for candidate in candidates:
        if not candidate.exists():
            continue
        if candidate.is_dir():
            if all((candidate / marker).exists() for marker in S_BSST265_MARKERS):
                return candidate
            dataset_dir = candidate / "dataset"
            if all((dataset_dir / marker).exists() for marker in S_BSST265_MARKERS):
                return dataset_dir
    return None


def build_s_bsst265_index(dataset_root: str | Path | None = None) -> pd.DataFrame:
    root = discover_s_bsst265_root(dataset_root)
    if root is None:
        raise FileNotFoundError("Could not locate S-BSST265 dataset root.")

    rows: list[dict[str, object]] = []
    with (root / "image_description.csv").open(encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle, delimiter=";")
        for row in reader:
            image_name = row["Image_Name"]
            mask_path = root / "groundtruth" / f"{image_name}.tif"
            image_path = root / "rawimages" / f"{image_name}.tif"
            with np.errstate(all="ignore"):
                mask = read_mask(mask_path)
            rows.append(
                {
                    "dataset": "s_bsst265",
                    "image_name": image_name,
                    "image_path": str(image_path),
                    "mask_path": str(mask_path),
                    "true_count": count_instances(mask),
                    "split": row["Train-/Testset split"],
                    "test_class": row["Testset class"],
                    "snr_class": row["S/N Class"],
                    "diagnosis": row["Diagnosis"],
                    "preparation": row["Preparation"],
                    "magnification": row["Magnification"],
                    "modality": row["Modality"],
                }
            )
    frame = pd.DataFrame(rows).sort_values("image_name", key=lambda series: series.map(natural_key)).reset_index(drop=True)
    return frame


def build_bbbc005_index(dataset_root: str | Path) -> pd.DataFrame:
    root = Path(dataset_root)
    rows = []
    for image_path in list_images(root):
        try:
            count = parse_count_from_filename(image_path.name)
        except ValueError:
            continue
        rows.append(
            {
                "dataset": "bbbc005",
                "image_name": image_path.stem,
                "image_path": str(image_path),
                "true_count": count,
                "split": "external",
            }
        )
    if not rows:
        raise FileNotFoundError(f"No parseable BBBC005 images found in {root}")
    return pd.DataFrame(rows).sort_values("image_name", key=lambda series: series.map(natural_key)).reset_index(drop=True)


def build_bbbc004_index(dataset_root: str | Path) -> pd.DataFrame:
    root = Path(dataset_root)
    rows = []
    for image_path in list_images(root):
        lower_name = image_path.name.lower()
        if any(token in lower_name for token in ("mask", "label", "seg", "groundtruth", "gt")):
            continue
        rows.append(
            {
                "dataset": "bbbc004",
                "image_name": image_path.stem,
                "image_path": str(image_path),
                "true_count": 300,
                "split": "external",
            }
        )
    if not rows:
        raise FileNotFoundError(f"No BBBC004 image files found in {root}")
    return pd.DataFrame(rows).sort_values("image_name", key=lambda series: series.map(natural_key)).reset_index(drop=True)


def read_mask(path: str | Path) -> np.ndarray:
    from tifffile import imread
    from PIL import Image

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return np.asarray(imread(path))
    except Exception:
        return np.asarray(Image.open(path))


def load_records_from_manifest(manifest_path: str | Path) -> pd.DataFrame:
    manifest = Path(manifest_path)
    frame = pd.read_csv(manifest)
    required_columns = {"image_path", "count"}
    missing = required_columns.difference(frame.columns)
    if missing:
        raise ValueError(f"Manifest is missing required columns: {sorted(missing)}")
    for column in ("image_path", "density_path", "mask_path", "points_path"):
        if column in frame.columns:
            frame[column] = frame[column].map(
                lambda value: str(Path(value).resolve())
                if isinstance(value, str) and Path(value).exists()
                else str((manifest.parent / value).resolve())
                if isinstance(value, str) and not Path(value).is_absolute()
                else value
            )
    if "true_count" not in frame.columns:
        frame["true_count"] = frame["count"]
    if "image_name" not in frame.columns:
        frame["image_name"] = frame["image_path"].map(lambda value: Path(value).stem)
    if "split" not in frame.columns:
        frame["split"] = "train"
    if "dataset" not in frame.columns:
        frame["dataset"] = "synthetic"
    return frame


def load_dataset_records(
    manifest_path: str | Path | None = None,
    dataset_name: str | None = None,
    dataset_root: str | Path | None = None,
) -> pd.DataFrame:
    if manifest_path is not None:
        return load_records_from_manifest(manifest_path)
    if dataset_name is None:
        raise ValueError("Either a manifest path or a dataset name must be provided.")
    dataset = dataset_name.lower()
    if dataset == "s_bsst265":
        return build_s_bsst265_index(dataset_root)
    if dataset == "bbbc004":
        if dataset_root is None:
            raise ValueError("BBBC004 evaluation requires --dataset-root.")
        return build_bbbc004_index(dataset_root)
    if dataset == "bbbc005":
        if dataset_root is None:
            raise ValueError("BBBC005 evaluation requires --dataset-root.")
        return build_bbbc005_index(dataset_root)
    raise ValueError(f"Unsupported dataset: {dataset_name}")


def export_frame(frame: pd.DataFrame, output_path: str | Path) -> Path:
    destination = Path(output_path)
    ensure_dir(destination.parent)
    frame.to_csv(destination, index=False)
    return destination


class DensityDataset(Dataset):
    def __init__(self, records: pd.DataFrame, augment: bool = False) -> None:
        self.records = records.reset_index(drop=True)
        self.augment = augment

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.records.iloc[index]
        image = read_image(row["image_path"]).astype(np.float32)
        if "density_path" in row and isinstance(row["density_path"], str) and row["density_path"]:
            density = np.load(row["density_path"]).astype(np.float32)
        elif "mask_path" in row and isinstance(row["mask_path"], str) and row["mask_path"]:
            density = density_from_mask(read_mask(row["mask_path"]))
        else:
            raise ValueError("Each training row must contain either density_path or mask_path.")

        if self.augment:
            if np.random.rand() < 0.5:
                image = np.fliplr(image).copy()
                density = np.fliplr(density).copy()
            if np.random.rand() < 0.5:
                image = np.flipud(image).copy()
                density = np.flipud(density).copy()

        image_tensor = torch.from_numpy(image[None, ...])
        density_tensor = torch.from_numpy(density[None, ...])
        return image_tensor, density_tensor


def split_records(records: pd.DataFrame, train_split: str = "train", val_split: str = "val") -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = records.copy()
    if val_split in set(frame["split"]):
        train_records = frame[frame["split"] == train_split].reset_index(drop=True)
        val_records = frame[frame["split"] == val_split].reset_index(drop=True)
        return train_records, val_records

    sorted_frame = frame.sort_values("image_name", key=lambda series: series.map(natural_key)).reset_index(drop=True)
    cutoff = max(1, int(round(len(sorted_frame) * 0.8)))
    train_records = sorted_frame.iloc[:cutoff].reset_index(drop=True)
    val_records = sorted_frame.iloc[cutoff:].reset_index(drop=True)
    if len(val_records) == 0:
        val_records = train_records.tail(1).copy()
        train_records = train_records.iloc[:-1].copy()
    return train_records.reset_index(drop=True), val_records.reset_index(drop=True)
