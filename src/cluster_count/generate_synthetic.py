from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage

from .common import ensure_dir, set_seed, write_png
from .data import export_frame, gaussian_density_from_points
from .visualization import save_overlay


PRESETS = {
    "default": {
        "min_cells": 8,
        "max_cells": 120,
        "blur_min": 0.4,
        "blur_max": 3.0,
        "cluster_min": 0.05,
        "cluster_max": 0.95,
        "intensity_mode": "normalized",
        "amplitude_min": 0.35,
        "amplitude_max": 0.90,
        "noise_base": 0.018,
        "noise_blur_scale": 0.010,
        "exposure_jitter": 0.00,
        "background_mean": 0.04,
        "background_std": 0.01,
        "background_gradient": 0.03,
        "cluster_radius_base": 0.03,
        "cluster_radius_scale": 0.06,
        "cell_sigma_min": 1.6,
        "cell_sigma_max": 3.8,
    },
    "operating_v2": {
        "min_cells": 10,
        "max_cells": 55,
        "blur_min": 0.4,
        "blur_max": 1.4,
        "cluster_min": 0.05,
        "cluster_max": 0.55,
        "intensity_mode": "fixed_exposure",
        "amplitude_min": 0.45,
        "amplitude_max": 0.78,
        "noise_base": 0.007,
        "noise_blur_scale": 0.003,
        "exposure_jitter": 0.04,
        "background_mean": 0.03,
        "background_std": 0.005,
        "background_gradient": 0.00,
        "cluster_radius_base": 0.03,
        "cluster_radius_scale": 0.045,
        "cell_sigma_min": 1.4,
        "cell_sigma_max": 2.6,
    },
    "stress": {
        "min_cells": 10,
        "max_cells": 80,
        "blur_min": 0.4,
        "blur_max": 1.6,
        "cluster_min": 0.05,
        "cluster_max": 0.60,
        "intensity_mode": "fixed_exposure",
        "amplitude_min": 0.42,
        "amplitude_max": 0.78,
        "noise_base": 0.008,
        "noise_blur_scale": 0.004,
        "exposure_jitter": 0.05,
        "background_mean": 0.03,
        "background_std": 0.005,
        "background_gradient": 0.00,
        "cluster_radius_base": 0.03,
        "cluster_radius_scale": 0.05,
        "cell_sigma_min": 1.4,
        "cell_sigma_max": 2.8,
    },
}


def sample_points(
    rng: np.random.Generator,
    count: int,
    shape: tuple[int, int],
    cluster_strength: float,
    cluster_radius_base: float = 0.03,
    cluster_radius_scale: float = 0.06,
) -> np.ndarray:
    height, width = shape
    margin = 8
    n_clusters = max(1, int(round(cluster_strength * max(1, count) / 8)))
    anchors = rng.uniform(low=(margin, margin), high=(height - margin, width - margin), size=(n_clusters, 2))
    cluster_radius = max(3.5, min(shape) * (cluster_radius_base + cluster_radius_scale * cluster_strength))

    points = []
    for _ in range(count):
        if rng.random() < cluster_strength:
            anchor = anchors[rng.integers(0, len(anchors))]
            point = anchor + rng.normal(loc=0.0, scale=cluster_radius, size=2)
        else:
            point = rng.uniform(low=(margin, margin), high=(height - margin, width - margin), size=2)
        point[0] = np.clip(point[0], margin, height - margin)
        point[1] = np.clip(point[1], margin, width - margin)
        points.append(point)
    return np.asarray(points, dtype=np.float32)


def render_cells(
    rng: np.random.Generator,
    points: np.ndarray,
    shape: tuple[int, int],
    blur_sigma: float,
    amplitude_range: tuple[float, float] = (0.35, 0.9),
    intensity_mode: str = "normalized",
    background_mean: float = 0.04,
    background_std: float = 0.01,
    noise_base: float = 0.018,
    noise_blur_scale: float = 0.010,
    exposure_jitter: float = 0.0,
    background_gradient: float = 0.03,
    cell_sigma_min: float = 1.6,
    cell_sigma_max: float = 3.8,
) -> np.ndarray:
    image = rng.normal(loc=background_mean, scale=background_std, size=shape).astype(np.float32)
    if background_gradient != 0.0:
        image += np.linspace(0.0, background_gradient, shape[1], dtype=np.float32)[None, :]

    for y, x in points:
        sigma = float(rng.uniform(cell_sigma_min, cell_sigma_max))
        radius = int(round(3 * sigma))
        yy, xx = np.mgrid[-radius : radius + 1, -radius : radius + 1]
        kernel = np.exp(-((xx ** 2 + yy ** 2) / (2.0 * sigma ** 2))).astype(np.float32)
        amplitude = float(rng.uniform(amplitude_range[0], amplitude_range[1]))
        patch = amplitude * kernel

        iy = int(round(float(y)))
        ix = int(round(float(x)))
        y0 = max(0, iy - radius)
        y1 = min(shape[0], iy + radius + 1)
        x0 = max(0, ix - radius)
        x1 = min(shape[1], ix + radius + 1)
        ky0 = y0 - (iy - radius)
        ky1 = ky0 + (y1 - y0)
        kx0 = x0 - (ix - radius)
        kx1 = kx0 + (x1 - x0)
        image[y0:y1, x0:x1] += patch[ky0:ky1, kx0:kx1]

    image = ndimage.gaussian_filter(image, sigma=blur_sigma)
    image += rng.normal(loc=0.0, scale=noise_base + noise_blur_scale * blur_sigma, size=shape).astype(np.float32)
    if intensity_mode == "fixed_exposure":
        if exposure_jitter > 0.0:
            image *= float(rng.uniform(1.0 - exposure_jitter, 1.0 + exposure_jitter))
        image = np.clip(image, 0.0, 1.0)
    else:
        image = image - image.min()
        image = image / max(float(image.max()), 1e-6)
    return image.astype(np.float32)


def build_record(
    base_dir: Path,
    image_path: Path,
    density_path: Path,
    points_path: Path,
    count: int,
    blur_sigma: float,
    cluster_strength: float,
    split: str,
) -> dict[str, object]:
    return {
        "image_name": image_path.stem,
        "image_path": str(image_path.relative_to(base_dir)),
        "density_path": str(density_path.relative_to(base_dir)),
        "points_path": str(points_path.relative_to(base_dir)),
        "count": int(count),
        "true_count": int(count),
        "blur_sigma": round(float(blur_sigma), 4),
        "cluster_strength": round(float(cluster_strength), 4),
        "split": split,
        "dataset": "synthetic",
    }


def generate_dataset(
    output_dir: str | Path,
    num_images: int = 64,
    image_size: int = 128,
    min_cells: int = 8,
    max_cells: int = 120,
    seed: int = 7,
    blur_min: float = 0.4,
    blur_max: float = 3.0,
    cluster_min: float = 0.05,
    cluster_max: float = 0.95,
    intensity_mode: str = "normalized",
    amplitude_min: float = 0.35,
    amplitude_max: float = 0.90,
    noise_base: float = 0.018,
    noise_blur_scale: float = 0.010,
    exposure_jitter: float = 0.0,
    background_mean: float = 0.04,
    background_std: float = 0.01,
    background_gradient: float = 0.03,
    cluster_radius_base: float = 0.03,
    cluster_radius_scale: float = 0.06,
    cell_sigma_min: float = 1.6,
    cell_sigma_max: float = 3.8,
) -> pd.DataFrame:
    set_seed(seed)
    rng = np.random.default_rng(seed)
    base = ensure_dir(output_dir)
    image_dir = ensure_dir(base / "images")
    density_dir = ensure_dir(base / "densities")
    points_dir = ensure_dir(base / "points")

    rows = []
    for index in range(num_images):
        count = int(rng.integers(min_cells, max_cells + 1))
        blur_sigma = float(rng.uniform(blur_min, blur_max))
        cluster_strength = float(rng.uniform(cluster_min, cluster_max))
        points = sample_points(
            rng,
            count=count,
            shape=(image_size, image_size),
            cluster_strength=cluster_strength,
            cluster_radius_base=cluster_radius_base,
            cluster_radius_scale=cluster_radius_scale,
        )
        density = gaussian_density_from_points(points, shape=(image_size, image_size), sigma=2.0)
        image = render_cells(
            rng,
            points=points,
            shape=(image_size, image_size),
            blur_sigma=blur_sigma,
            amplitude_range=(amplitude_min, amplitude_max),
            intensity_mode=intensity_mode,
            background_mean=background_mean,
            background_std=background_std,
            noise_base=noise_base,
            noise_blur_scale=noise_blur_scale,
            exposure_jitter=exposure_jitter,
            background_gradient=background_gradient,
            cell_sigma_min=cell_sigma_min,
            cell_sigma_max=cell_sigma_max,
        )

        split = "train"
        if index >= int(0.8 * num_images):
            split = "val"
        if index >= int(0.9 * num_images):
            split = "test"

        image_path = image_dir / f"synthetic_{index:04d}.png"
        density_path = density_dir / f"synthetic_{index:04d}.npy"
        points_path = points_dir / f"synthetic_{index:04d}.npy"
        write_png(image_path, image)
        np.save(density_path, density.astype(np.float32))
        np.save(points_path, points.astype(np.float32))
        rows.append(
            build_record(
                base_dir=base,
                image_path=image_path,
                density_path=density_path,
                points_path=points_path,
                count=count,
                blur_sigma=blur_sigma,
                cluster_strength=cluster_strength,
                split=split,
            )
        )

    frame = pd.DataFrame(rows)
    export_frame(frame, base / "manifest.csv")

    preview_row = frame.iloc[0]
    preview_image = plt_safe_read(base / preview_row["image_path"])
    preview_points = np.load(base / preview_row["points_path"])
    save_overlay(
        preview_image,
        base / "preview_overlay.png",
        points=preview_points,
        title="Synthetic preview",
        annotations=[
            f"count={int(preview_row['count'])}",
            f"blur_sigma={preview_row['blur_sigma']}",
            f"cluster_strength={preview_row['cluster_strength']}",
        ],
    )
    return frame


def plt_safe_read(path: str | Path) -> np.ndarray:
    from .common import read_image

    return read_image(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate synthetic clustered microscopy images with known counts.")
    parser.add_argument("--output", type=Path, required=True, help="Output directory for synthetic images and manifest.")
    parser.add_argument("--num-images", type=int, default=64, help="Number of synthetic images to create.")
    parser.add_argument("--image-size", type=int, default=128, help="Square image size in pixels.")
    parser.add_argument(
        "--preset",
        choices=sorted(PRESETS),
        default="default",
        help="Synthetic benchmark preset. Use operating_v2 for the calibrated operating-range benchmark.",
    )
    parser.add_argument("--min-cells", type=int, default=8, help="Minimum count per image.")
    parser.add_argument("--max-cells", type=int, default=120, help="Maximum count per image.")
    parser.add_argument("--blur-min", type=float, default=None, help="Minimum Gaussian blur sigma.")
    parser.add_argument("--blur-max", type=float, default=None, help="Maximum Gaussian blur sigma.")
    parser.add_argument("--cluster-min", type=float, default=None, help="Minimum clustering strength.")
    parser.add_argument("--cluster-max", type=float, default=None, help="Maximum clustering strength.")
    parser.add_argument(
        "--intensity-mode",
        choices=["normalized", "fixed_exposure"],
        default=None,
        help="Per-image normalization or fixed-exposure clipping.",
    )
    parser.add_argument("--seed", type=int, default=7, help="Random seed.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    preset = PRESETS[args.preset]
    frame = generate_dataset(
        output_dir=args.output,
        num_images=args.num_images,
        image_size=args.image_size,
        min_cells=args.min_cells if args.min_cells != 8 or args.preset == "default" else preset["min_cells"],
        max_cells=args.max_cells if args.max_cells != 120 or args.preset == "default" else preset["max_cells"],
        seed=args.seed,
        blur_min=args.blur_min if args.blur_min is not None else preset["blur_min"],
        blur_max=args.blur_max if args.blur_max is not None else preset["blur_max"],
        cluster_min=args.cluster_min if args.cluster_min is not None else preset["cluster_min"],
        cluster_max=args.cluster_max if args.cluster_max is not None else preset["cluster_max"],
        intensity_mode=args.intensity_mode if args.intensity_mode is not None else preset["intensity_mode"],
        amplitude_min=preset["amplitude_min"],
        amplitude_max=preset["amplitude_max"],
        noise_base=preset["noise_base"],
        noise_blur_scale=preset["noise_blur_scale"],
        exposure_jitter=preset["exposure_jitter"],
        background_mean=preset["background_mean"],
        background_std=preset["background_std"],
        background_gradient=preset["background_gradient"],
        cluster_radius_base=preset["cluster_radius_base"],
        cluster_radius_scale=preset["cluster_radius_scale"],
        cell_sigma_min=preset["cell_sigma_min"],
        cell_sigma_max=preset["cell_sigma_max"],
    )
    print(f"Generated {len(frame)} synthetic images at {args.output}")


if __name__ == "__main__":
    main()
