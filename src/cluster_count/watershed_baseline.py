from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy import ndimage
from skimage.feature import peak_local_max
from skimage.filters import threshold_otsu
from skimage.segmentation import watershed

from .common import ensure_dir, read_image
from .visualization import save_overlay


def _remove_small_components(binary: np.ndarray, min_area: int = 12) -> np.ndarray:
    labels, num_labels = ndimage.label(binary)
    if num_labels == 0:
        return binary
    areas = np.bincount(labels.ravel())
    keep = areas >= min_area
    keep[0] = False
    return keep[labels]


def run_watershed(image: np.ndarray) -> dict[str, object]:
    smooth = ndimage.gaussian_filter(image.astype(np.float32), sigma=1.0)
    threshold = threshold_otsu(smooth)
    binary = smooth > threshold
    binary = _remove_small_components(binary, min_area=12)

    if int(binary.sum()) == 0:
        return {
            "count": 0.0,
            "labels": np.zeros_like(binary, dtype=np.int32),
            "points": np.empty((0, 2), dtype=np.int32),
            "connected_components": 0,
        }

    distance = ndimage.distance_transform_edt(binary)
    peak_coords = peak_local_max(distance, min_distance=4, labels=binary, exclude_border=False)
    markers = np.zeros(binary.shape, dtype=np.int32)
    if len(peak_coords) == 0:
        peak = np.unravel_index(np.argmax(distance), distance.shape)
        peak_coords = np.asarray([[int(peak[0]), int(peak[1])]], dtype=np.int32)
    for index, (y, x) in enumerate(peak_coords, start=1):
        markers[int(y), int(x)] = index
    labels = watershed(-distance, markers=markers, mask=binary)
    count = int(labels.max())
    centers = ndimage.center_of_mass(binary, labels=labels, index=range(1, count + 1))
    points = np.asarray(centers, dtype=np.float32) if count > 0 else np.empty((0, 2), dtype=np.float32)
    connected_components = int(ndimage.label(binary)[1])
    return {
        "count": float(count),
        "labels": labels,
        "points": points,
        "connected_components": connected_components,
        "foreground_fraction": float(binary.mean()),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a watershed counting baseline on a microscopy image.")
    parser.add_argument("--image", type=Path, required=True, help="Input microscopy image.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for overlay output.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    ensure_dir(args.output_dir)
    image = read_image(args.image)
    result = run_watershed(image)
    save_overlay(
        image,
        args.output_dir / f"{args.image.stem}_watershed_overlay.png",
        points=np.asarray(result["points"]),
        title=f"Watershed baseline count={result['count']:.1f}",
    )
    print(f"Watershed count for {args.image.name}: {result['count']:.1f}")


if __name__ == "__main__":
    main()
