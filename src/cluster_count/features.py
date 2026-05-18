from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage
from skimage.feature import peak_local_max

from .common import ensure_dir, read_image


FEATURE_THRESHOLDS = (0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25)
PEAK_SIGMAS = (0.5, 1.0, 1.5)
PEAK_PERCENTILES = (90, 94, 96, 98)


def extract_count_features(image: np.ndarray) -> dict[str, float]:
    array = np.asarray(image, dtype=np.float32)
    smooth1 = ndimage.gaussian_filter(array, 1.0)
    smooth2 = ndimage.gaussian_filter(array, 2.0)
    grad_y, grad_x = np.gradient(array)
    features: dict[str, float] = {
        "sum": float(array.sum()),
        "mean": float(array.mean()),
        "std": float(array.std()),
        "p90": float(np.percentile(array, 90)),
        "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)),
        "lap_var": float(ndimage.laplace(array).var()),
        "grad_mean": float(np.hypot(grad_y, grad_x).mean()),
        "smooth1_sum": float(smooth1.sum()),
        "smooth2_sum": float(smooth2.sum()),
    }

    for threshold in FEATURE_THRESHOLDS:
        binary = array > threshold
        features[f"fg_{threshold:.2f}"] = float(binary.mean())
        features[f"cc_{threshold:.2f}"] = float(ndimage.label(binary)[1])

    for sigma in PEAK_SIGMAS:
        smooth = ndimage.gaussian_filter(array, sigma)
        for percentile in PEAK_PERCENTILES:
            threshold = float(np.percentile(smooth, percentile))
            peaks = peak_local_max(smooth, min_distance=2, threshold_abs=threshold, exclude_border=False)
            features[f"pk_s{sigma:.1f}_p{percentile}"] = float(len(peaks))

    return features


def build_feature_frame(records: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for _, row in records.iterrows():
        image = read_image(row["image_path"])
        features = extract_count_features(image)
        record = row.to_dict()
        record.update(features)
        rows.append(record)
    return pd.DataFrame(rows)


def export_feature_frame(frame: pd.DataFrame, output_path: str | Path) -> Path:
    destination = Path(output_path)
    ensure_dir(destination.parent)
    frame.to_csv(destination, index=False)
    return destination

