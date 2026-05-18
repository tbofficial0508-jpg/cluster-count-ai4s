from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from skimage.feature import peak_local_max

from .common import clamp, ensure_dir


def find_density_peaks(
    density_map: np.ndarray,
    expected_count: float | None = None,
    min_distance: int = 4,
) -> np.ndarray:
    density = np.asarray(density_map, dtype=np.float32)
    if density.size == 0 or float(density.max()) <= 0.0:
        return np.empty((0, 2), dtype=np.int32)
    smooth = ndimage.gaussian_filter(density, sigma=1.0)
    threshold = max(float(np.percentile(smooth, 98)), float(smooth.mean() + smooth.std()))
    coords = peak_local_max(
        smooth,
        min_distance=min_distance,
        threshold_abs=threshold * 0.15,
        exclude_border=False,
    )
    if expected_count is not None and len(coords) > int(round(expected_count)):
        scores = smooth[coords[:, 0], coords[:, 1]]
        order = np.argsort(scores)[::-1][: int(round(expected_count))]
        coords = coords[order]
    if coords.size == 0:
        peak = np.unravel_index(np.argmax(smooth), smooth.shape)
        coords = np.asarray([[int(peak[0]), int(peak[1])]], dtype=np.int32)
    return np.asarray(coords, dtype=np.int32)


def save_overlay(
    image: np.ndarray,
    output_path: str | Path,
    points: np.ndarray | None = None,
    title: str | None = None,
    annotations: list[str] | None = None,
) -> Path:
    destination = Path(output_path)
    ensure_dir(destination.parent)
    figure, axis = plt.subplots(figsize=(7, 7))
    axis.imshow(image, cmap="gray")
    if points is not None and len(points) > 0:
        axis.scatter(points[:, 1], points[:, 0], s=20, c="#00e676", edgecolors="black", linewidths=0.4)
    axis.set_axis_off()

    lines = []
    if title:
        lines.append(title)
    if annotations:
        lines.extend([item for item in annotations if item])
    if lines:
        axis.set_title("\n".join(lines), fontsize=10)

    figure.tight_layout()
    figure.savefig(destination, dpi=160, bbox_inches="tight")
    plt.close(figure)
    return destination


def estimate_blur_score(image: np.ndarray) -> float:
    lap_var = float(ndimage.laplace(image).var())
    focus = clamp(lap_var / 0.02, 0.0, 1.0)
    return 1.0 - focus

