from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .calibration import (
    build_patchwise_calibration_features,
    compute_calibrator_uncertainty,
    load_patchwise_calibrator_bundle,
)
from .center_heatmap import load_center_checkpoint, predict_patchwise_center_heatmap
from .common import clamp, ensure_dir, format_warnings, read_image, select_device
from .data import load_dataset_records
from .modeling import DensityCountingCNN, density_to_count, load_checkpoint
from .patchwise import predict_patchwise_density_map
from .visualization import estimate_blur_score, find_density_peaks, save_overlay
from .watershed_baseline import run_watershed


def _enable_mc_dropout(model: torch.nn.Module) -> None:
    model.eval()
    for module in model.modules():
        if isinstance(module, (torch.nn.Dropout, torch.nn.Dropout2d, torch.nn.Dropout3d)):
            module.train()


def predict_density_map_with_model(
    image: np.ndarray,
    model: DensityCountingCNN,
    device: torch.device,
    mc_samples: int = 8,
) -> dict[str, object]:
    image_tensor = torch.from_numpy(image[None, None, ...]).to(device=device, dtype=torch.float32)

    predictions = []
    with torch.no_grad():
        for _ in range(max(mc_samples, 1)):
            if mc_samples > 1:
                _enable_mc_dropout(model)
            else:
                model.eval()
            predictions.append(model(image_tensor).cpu())

    stack = torch.stack(predictions, dim=0)
    mean_density = stack.mean(dim=0)
    count_samples = density_to_count(stack).cpu().numpy().reshape(-1)
    return {
        "density_map": mean_density.squeeze().numpy(),
        "count": float(count_samples.mean()),
        "count_std": float(count_samples.std(ddof=0)),
    }


def predict_density_map(
    image: np.ndarray,
    checkpoint_path: str | Path,
    device: torch.device,
    mc_samples: int = 8,
) -> dict[str, object]:
    model, checkpoint = load_checkpoint(checkpoint_path, device=device)
    result = predict_density_map_with_model(image=image, model=model, device=device, mc_samples=mc_samples)
    result["checkpoint"] = checkpoint
    return result


def score_prediction(
    image: np.ndarray,
    pred_count: float,
    baseline_count: float,
    count_std: float,
    connected_components: int,
) -> dict[str, float]:
    blur_score = estimate_blur_score(image)
    disagreement = abs(pred_count - baseline_count)
    disagreement_ratio = disagreement / max(pred_count, baseline_count, 1.0)
    cluster_score = clamp((pred_count - connected_components) / max(pred_count, 1.0), 0.0, 1.0)
    std_ratio = count_std / max(pred_count, 1.0)
    confidence = clamp(1.0 - (0.45 * std_ratio + 0.30 * disagreement_ratio + 0.15 * blur_score + 0.10 * cluster_score))
    return {
        "blur_score": float(blur_score),
        "disagreement": float(disagreement),
        "disagreement_ratio": float(disagreement_ratio),
        "cluster_score": float(cluster_score),
        "confidence": float(confidence),
    }


def collect_warnings(metrics: dict[str, float]) -> list[str]:
    warnings: list[str] = []
    if metrics["cluster_score"] > 0.35:
        warnings.append("high clustering")
    if metrics["blur_score"] > 0.55:
        warnings.append("strong blur or low focus")
    if metrics["disagreement_ratio"] > 0.25:
        warnings.append("model-baseline disagreement")
    if metrics["confidence"] < 0.55:
        warnings.append("low confidence")
    return warnings


def run_prediction_table(
    records: pd.DataFrame,
    output_dir: str | Path,
    checkpoint_path: str | Path | None = None,
    method: str = "ensemble",
    device: str | torch.device = "cpu",
    mc_samples: int = 8,
    save_overlays: bool = True,
) -> pd.DataFrame:
    output_base = ensure_dir(output_dir)
    overlay_dir = ensure_dir(output_base / "overlays")
    device_obj = select_device(device) if isinstance(device, str) else torch.device(device)
    rows: list[dict[str, object]] = []
    loaded_model: DensityCountingCNN | None = None
    loaded_center_model = None
    loaded_checkpoint: dict[str, object] | None = None
    loaded_calibrator_bundle: dict[str, object] | None = None
    if checkpoint_path is not None and method in {"ensemble", "density-cnn", "patchwise-cnn"}:
        loaded_model, loaded_checkpoint = load_checkpoint(checkpoint_path, device=device_obj)
    elif checkpoint_path is not None and method == "center-unet":
        loaded_center_model, loaded_checkpoint = load_center_checkpoint(checkpoint_path, device=device_obj)
    elif checkpoint_path is not None and method == "patchwise-calibrated":
        loaded_calibrator_bundle = load_patchwise_calibrator_bundle(checkpoint_path)
        patchwise_checkpoint = loaded_calibrator_bundle["patchwise_checkpoint"]
        loaded_model, loaded_checkpoint = load_checkpoint(patchwise_checkpoint, device=device_obj)

    for _, row in records.iterrows():
        image = read_image(row["image_path"])
        baseline = run_watershed(image)
        baseline_count = float(baseline["count"])
        connected_components = int(baseline.get("connected_components", 0))

        density_result: dict[str, object] | None = None
        raw_pred_count: float | None = None
        raw_count_std: float | None = None
        if method in {"ensemble", "density-cnn"} and loaded_model is not None:
            density_result = predict_density_map_with_model(image=image, model=loaded_model, device=device_obj, mc_samples=mc_samples)
            pred_count = float(density_result["count"])
            count_std = float(density_result["count_std"])
            peak_points = find_density_peaks(np.asarray(density_result["density_map"]), expected_count=pred_count)
        elif method == "patchwise-cnn" and loaded_model is not None:
            checkpoint_reference_mag = float(loaded_checkpoint.get("reference_magnification", 40.0)) if loaded_checkpoint else 40.0
            checkpoint_patch_size = int(loaded_checkpoint.get("patch_size", 256)) if loaded_checkpoint else 256
            checkpoint_patch_stride = int(loaded_checkpoint.get("patch_stride", 224)) if loaded_checkpoint else 224
            checkpoint_input_normalization = str(loaded_checkpoint.get("input_normalization", "zscore")) if loaded_checkpoint else "zscore"
            density_result = predict_patchwise_density_map(
                model=loaded_model,
                image=image,
                device=device_obj,
                magnification=row.get("magnification"),
                reference_magnification=float(row.get("reference_magnification", checkpoint_reference_mag)),
                min_scale=float(row.get("min_scale", 0.50)),
                max_scale=float(row.get("max_scale", 4.00)),
                patch_size=int(row.get("patch_size", checkpoint_patch_size)),
                stride=int(row.get("patch_stride", checkpoint_patch_stride)),
                mc_samples=mc_samples,
                batch_size=int(row.get("batch_size", 4)),
                input_normalization=str(row.get("input_normalization", checkpoint_input_normalization)),
            )
            pred_count = float(density_result["count"])
            count_std = float(density_result["count_std"])
            peak_points = np.asarray(density_result["peak_points"])
        elif method == "center-unet" and loaded_center_model is not None:
            checkpoint_reference_mag = float(loaded_checkpoint.get("reference_magnification", 40.0)) if loaded_checkpoint else 40.0
            checkpoint_patch_size = int(loaded_checkpoint.get("patch_size", 192)) if loaded_checkpoint else 192
            checkpoint_patch_stride = int(loaded_checkpoint.get("patch_stride", 160)) if loaded_checkpoint else 160
            checkpoint_input_normalization = str(loaded_checkpoint.get("input_normalization", "zscore")) if loaded_checkpoint else "zscore"
            checkpoint_peak_threshold = float(loaded_checkpoint.get("peak_threshold", 0.35)) if loaded_checkpoint else 0.35
            checkpoint_peak_min_distance = int(loaded_checkpoint.get("peak_min_distance", 3)) if loaded_checkpoint else 3
            density_result = predict_patchwise_center_heatmap(
                model=loaded_center_model,
                image=image,
                device=device_obj,
                magnification=row.get("magnification"),
                reference_magnification=float(row.get("reference_magnification", checkpoint_reference_mag)),
                min_scale=float(row.get("min_scale", 0.50)),
                max_scale=float(row.get("max_scale", 4.00)),
                patch_size=int(row.get("patch_size", checkpoint_patch_size)),
                stride=int(row.get("patch_stride", checkpoint_patch_stride)),
                mc_samples=mc_samples,
                batch_size=int(row.get("batch_size", 4)),
                input_normalization=str(row.get("input_normalization", checkpoint_input_normalization)),
                peak_threshold=float(row.get("peak_threshold", checkpoint_peak_threshold)),
                peak_min_distance=int(row.get("peak_min_distance", checkpoint_peak_min_distance)),
            )
            pred_count = float(density_result["count"])
            count_std = float(density_result["count_std"])
            peak_points = np.asarray(density_result["peak_points"])
        elif method == "patchwise-calibrated" and loaded_model is not None and loaded_calibrator_bundle is not None:
            patchwise_config = loaded_calibrator_bundle.get("patchwise_config", {})
            density_result = predict_patchwise_density_map(
                model=loaded_model,
                image=image,
                device=device_obj,
                magnification=row.get("magnification"),
                reference_magnification=float(row.get("reference_magnification", patchwise_config.get("reference_magnification", 40.0))),
                min_scale=float(row.get("min_scale", patchwise_config.get("min_scale", 0.50))),
                max_scale=float(row.get("max_scale", patchwise_config.get("max_scale", 4.00))),
                patch_size=int(row.get("patch_size", patchwise_config.get("patch_size", 192))),
                stride=int(row.get("patch_stride", patchwise_config.get("patch_stride", 160))),
                mc_samples=mc_samples,
                batch_size=int(row.get("batch_size", patchwise_config.get("batch_size", 8))),
                input_normalization=str(row.get("input_normalization", patchwise_config.get("input_normalization", "zscore"))),
            )
            raw_pred_count = float(density_result["count"])
            raw_count_std = float(density_result["count_std"])
            peak_points = np.asarray(density_result["peak_points"])

            raw_metrics = score_prediction(
                image=image,
                pred_count=raw_pred_count,
                baseline_count=baseline_count,
                count_std=raw_count_std,
                connected_components=connected_components,
            )
            calibration_features = build_patchwise_calibration_features(
                pred_count=raw_pred_count,
                baseline_count=baseline_count,
                count_std=raw_count_std,
                disagreement=raw_metrics["disagreement"],
                confidence=raw_metrics["confidence"],
            )
            calibrator_model = loaded_calibrator_bundle["model"]
            calibrator_mean, calibrator_std = compute_calibrator_uncertainty(
                calibrator_model,
                calibration_features[loaded_calibrator_bundle["feature_columns"]],
            )
            pred_count = float(calibrator_mean[0])
            count_std = float(calibrator_std[0])
        else:
            pred_count = baseline_count
            count_std = 0.0
            peak_points = np.asarray(baseline["points"])

        if method == "watershed":
            pred_count = baseline_count
            count_std = 0.0
            peak_points = np.asarray(baseline["points"])

        metrics = score_prediction(
            image=image,
            pred_count=pred_count,
            baseline_count=baseline_count,
            count_std=count_std,
            connected_components=connected_components,
        )
        warnings = collect_warnings(metrics)
        overlay_path = overlay_dir / f"{row['image_name']}_overlay.png"
        if save_overlays:
            annotations = [
                f"pred={pred_count:.1f} baseline={baseline_count:.1f} conf={metrics['confidence']:.2f}",
                f"uncertainty={count_std:.2f} blur={metrics['blur_score']:.2f} cluster={metrics['cluster_score']:.2f}",
                f"warnings={format_warnings(warnings) or 'none'}",
            ]
            if method == "patchwise-calibrated" and raw_pred_count is not None:
                annotations[0] = f"pred={pred_count:.1f} raw={raw_pred_count:.1f} baseline={baseline_count:.1f} conf={metrics['confidence']:.2f}"
            if "true_count" in row and not pd.isna(row["true_count"]):
                annotations.insert(1, f"truth={float(row['true_count']):.1f}")
            save_overlay(image, overlay_path, points=peak_points, title=str(row["image_name"]), annotations=annotations)

        result_row = row.to_dict()
        result_row.update(
            {
                "pred_count": float(pred_count),
                "baseline_count": float(baseline_count),
                "count_std": float(count_std),
                "blur_score": metrics["blur_score"],
                "cluster_score": metrics["cluster_score"],
                "disagreement": metrics["disagreement"],
                "disagreement_ratio": metrics["disagreement_ratio"],
                "confidence": metrics["confidence"],
                "warnings": format_warnings(warnings),
                "overlay_path": str(overlay_path),
                "prediction_method": method,
            }
        )
        if raw_pred_count is not None:
            result_row["raw_pred_count"] = float(raw_pred_count)
        if raw_count_std is not None:
            result_row["raw_count_std"] = float(raw_count_std)
        if density_result is not None and "scale_factor" in density_result:
            result_row["scale_factor"] = float(density_result["scale_factor"])
        if method == "patchwise-calibrated":
            result_row["model_family"] = "patchwise_calibrated"
        elif loaded_checkpoint is not None and "model_kwargs" in loaded_checkpoint:
            result_row["model_family"] = str(loaded_checkpoint.get("model_family", "density_cnn"))
        if "true_count" in result_row and result_row["true_count"] is not None and not pd.isna(result_row["true_count"]):
            truth = float(result_row["true_count"])
            result_row["abs_error"] = abs(pred_count - truth)
            result_row["baseline_abs_error"] = abs(baseline_count - truth)
        rows.append(result_row)

    frame = pd.DataFrame(rows)
    frame.to_csv(output_base / "count.csv", index=False)
    return frame


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Predict cell counts with a density CNN, watershed baseline, or both.")
    parser.add_argument("--image", type=Path, help="Single image path for ad hoc prediction.")
    parser.add_argument("--manifest", type=Path, help="Manifest CSV for batch prediction.")
    parser.add_argument("--dataset", type=str, help="Named dataset adapter: s_bsst265, bbbc004, or bbbc005.")
    parser.add_argument("--dataset-root", type=Path, help="Dataset root for named dataset adapters.")
    parser.add_argument("--model-checkpoint", type=Path, help="Trained density CNN checkpoint.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for count.csv and overlays.")
    parser.add_argument("--method", choices=["ensemble", "density-cnn", "patchwise-cnn", "patchwise-calibrated", "center-unet", "watershed"], default="ensemble")
    parser.add_argument("--limit", type=int, default=0, help="Optional record limit.")
    parser.add_argument("--split", type=str, help="Optional split filter such as train, val, or test.")
    parser.add_argument("--mc-samples", type=int, default=8, help="MC dropout samples for uncertainty estimation.")
    parser.add_argument("--device", type=str, default="cpu", help="Inference device.")
    parser.add_argument("--patch-size", type=int, default=256, help="Patch size for patchwise-cnn inference.")
    parser.add_argument("--patch-stride", type=int, default=224, help="Patch stride for patchwise-cnn inference.")
    parser.add_argument("--reference-magnification", type=float, default=40.0, help="Reference magnification for scale normalization.")
    parser.add_argument("--min-scale", type=float, default=0.50, help="Minimum scale factor after magnification normalization.")
    parser.add_argument("--max-scale", type=float, default=4.00, help="Maximum scale factor after magnification normalization.")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for patchwise-cnn inference.")
    parser.add_argument("--input-normalization", type=str, choices=["none", "zscore"], default="zscore", help="Patch intensity normalization mode for patchwise-cnn.")
    parser.add_argument("--peak-threshold", type=float, default=0.35, help="Peak threshold for center-unet inference.")
    parser.add_argument("--peak-min-distance", type=int, default=3, help="Peak min-distance for center-unet inference.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.image:
        records = pd.DataFrame(
            [
                {
                    "dataset": "custom",
                    "image_name": args.image.stem,
                    "image_path": str(args.image),
                    "split": "external",
                }
            ]
        )
    else:
        records = load_dataset_records(manifest_path=args.manifest, dataset_name=args.dataset, dataset_root=args.dataset_root)
        if args.split:
            records = records[records["split"] == args.split].reset_index(drop=True)

    if args.limit > 0:
        records = records.head(args.limit).reset_index(drop=True)
    records = records.copy()
    records["patch_size"] = args.patch_size
    records["patch_stride"] = args.patch_stride
    records["reference_magnification"] = args.reference_magnification
    records["min_scale"] = args.min_scale
    records["max_scale"] = args.max_scale
    records["batch_size"] = args.batch_size
    records["input_normalization"] = args.input_normalization
    records["peak_threshold"] = args.peak_threshold
    records["peak_min_distance"] = args.peak_min_distance

    frame = run_prediction_table(
        records=records,
        output_dir=args.output_dir,
        checkpoint_path=args.model_checkpoint,
        method=args.method,
        device=args.device,
        mc_samples=args.mc_samples,
        save_overlays=True,
    )
    print(f"Saved predictions for {len(frame)} image(s) to {args.output_dir}")


if __name__ == "__main__":
    main()
