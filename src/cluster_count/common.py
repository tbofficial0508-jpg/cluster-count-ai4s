from __future__ import annotations

import json
import logging
import math
import random
import re
import warnings
from pathlib import Path
from typing import Iterable

import numpy as np
import tifffile
from PIL import Image
import torch

tifffile_logger = logging.getLogger("tifffile")
tifffile_logger.setLevel(logging.ERROR)


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def ensure_dir(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def natural_key(value: str) -> list[object]:
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", value)]


def read_image(path: str | Path) -> np.ndarray:
    image_path = Path(path)
    suffix = image_path.suffix.lower()
    if suffix not in IMAGE_SUFFIXES:
        raise ValueError(f"Unsupported image suffix: {suffix}")
    if suffix in {".tif", ".tiff"}:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                array = tifffile.imread(image_path)
            except Exception:
                array = np.asarray(Image.open(image_path))
    else:
        array = np.asarray(Image.open(image_path))
    if array.ndim == 3:
        array = array.mean(axis=2)
    array = array.astype(np.float32)
    if array.max() > 1.0:
        array = array / 255.0
    return np.clip(array, 0.0, 1.0)


def write_png(path: str | Path, array: np.ndarray) -> Path:
    image_path = Path(path)
    ensure_dir(image_path.parent)
    data = np.asarray(array, dtype=np.float32)
    if data.ndim == 2:
        data = np.clip(data, 0.0, 1.0)
        out = (data * 255.0).round().astype(np.uint8)
        Image.fromarray(out, mode="L").save(image_path)
    else:
        data = np.clip(data, 0.0, 1.0)
        out = (data * 255.0).round().astype(np.uint8)
        Image.fromarray(out).save(image_path)
    return image_path


def save_json(path: str | Path, payload: dict) -> Path:
    output_path = Path(path)
    ensure_dir(output_path.parent)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output_path


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def list_images(root: str | Path) -> list[Path]:
    base = Path(root)
    return sorted(
        [path for path in base.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES],
        key=lambda item: natural_key(item.name),
    )


def rel_or_abs(path: str | Path, base: str | Path | None = None) -> str:
    target = Path(path)
    if base is None:
        return str(target)
    try:
        return str(target.relative_to(base))
    except ValueError:
        return str(target)


def normalize_density_map(density_map: np.ndarray, target_count: float) -> np.ndarray:
    density = np.asarray(density_map, dtype=np.float32)
    current_sum = float(density.sum())
    if current_sum <= 0.0:
        return density
    return density * (float(target_count) / current_sum)


def format_warnings(items: Iterable[str]) -> str:
    values = [item for item in items if item]
    return "; ".join(values)


def sigmoid_scale(value: float, midpoint: float, slope: float) -> float:
    return 1.0 / (1.0 + math.exp(-slope * (value - midpoint)))


def select_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)
