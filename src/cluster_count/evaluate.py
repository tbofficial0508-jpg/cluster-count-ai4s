from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .common import save_json
from .data import load_dataset_records
from .metrics import compute_count_metrics
from .predict import run_prediction_table


def summarise_grouped_metrics(frame: pd.DataFrame, group_column: str, prediction_column: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    if group_column not in frame.columns:
        return rows
    for value, group in frame.groupby(group_column, dropna=False):
        label = "missing" if pd.isna(value) else str(value)
        metrics = compute_count_metrics(group["true_count"], group[prediction_column])
        rows.append({"group": label, **metrics})
    return rows


def evaluate_predictions(frame: pd.DataFrame) -> dict[str, object]:
    metrics: dict[str, object] = {
        "n_images": int(len(frame)),
        "model": compute_count_metrics(frame["true_count"], frame["pred_count"]),
        "watershed": compute_count_metrics(frame["true_count"], frame["baseline_count"]),
        "hardest_examples": [],
    }
    hardest = frame.sort_values("abs_error", ascending=False).head(5)
    metrics["hardest_examples"] = hardest[
        ["image_name", "true_count", "pred_count", "baseline_count", "abs_error", "confidence", "warnings", "overlay_path"]
    ].to_dict(orient="records")

    if "dataset" in frame.columns and len(frame["dataset"].unique()) == 1:
        dataset_name = str(frame["dataset"].iloc[0])
        if dataset_name == "synthetic":
            blur_bins = pd.cut(frame["blur_sigma"], bins=[0.0, 1.0, 2.0, 10.0], labels=["low", "medium", "high"])
            cluster_bins = pd.cut(
                frame["cluster_strength"],
                bins=[0.0, 0.33, 0.66, 1.0],
                labels=["low", "medium", "high"],
                include_lowest=True,
            )
            synthetic_frame = frame.assign(blur_bucket=blur_bins, cluster_bucket=cluster_bins)
            metrics["failure_regimes"] = {
                "blur_bucket": summarise_grouped_metrics(synthetic_frame, "blur_bucket", "pred_count"),
                "cluster_bucket": summarise_grouped_metrics(synthetic_frame, "cluster_bucket", "pred_count"),
            }
        elif dataset_name == "s_bsst265":
            metrics["failure_regimes"] = {
                "snr_class": summarise_grouped_metrics(frame, "snr_class", "pred_count"),
                "test_class": summarise_grouped_metrics(frame, "test_class", "pred_count"),
                "magnification": summarise_grouped_metrics(frame, "magnification", "pred_count"),
                "modality": summarise_grouped_metrics(frame, "modality", "pred_count"),
                "preparation": summarise_grouped_metrics(frame, "preparation", "pred_count"),
                "diagnosis": summarise_grouped_metrics(frame, "diagnosis", "pred_count"),
            }
    return metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate counting predictions against known counts.")
    parser.add_argument("--manifest", type=Path, help="Synthetic or custom manifest CSV.")
    parser.add_argument("--dataset", type=str, help="Named dataset adapter: s_bsst265, bbbc004, or bbbc005.")
    parser.add_argument("--dataset-root", type=Path, help="Dataset root for named dataset adapters.")
    parser.add_argument("--model-checkpoint", type=Path, help="Density CNN checkpoint.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for predictions, metrics, and report.")
    parser.add_argument("--method", choices=["ensemble", "density-cnn", "patchwise-cnn", "patchwise-calibrated", "center-unet", "watershed"], default="ensemble")
    parser.add_argument("--limit", type=int, default=0, help="Optional record limit.")
    parser.add_argument("--split", type=str, help="Optional split filter.")
    parser.add_argument("--mc-samples", type=int, default=8, help="MC dropout samples.")
    parser.add_argument("--device", type=str, default="cpu", help="Inference device.")
    parser.add_argument("--patch-size", type=int, default=256, help="Patch size for patchwise-cnn evaluation.")
    parser.add_argument("--patch-stride", type=int, default=224, help="Patch stride for patchwise-cnn evaluation.")
    parser.add_argument("--reference-magnification", type=float, default=40.0, help="Reference magnification for scale normalization.")
    parser.add_argument("--min-scale", type=float, default=0.50, help="Minimum scale factor after magnification normalization.")
    parser.add_argument("--max-scale", type=float, default=4.00, help="Maximum scale factor after magnification normalization.")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for patchwise-cnn evaluation.")
    parser.add_argument("--input-normalization", type=str, choices=["none", "zscore"], default="zscore", help="Patch intensity normalization mode for patchwise-cnn evaluation.")
    parser.add_argument("--peak-threshold", type=float, default=0.35, help="Peak threshold for center-unet evaluation.")
    parser.add_argument("--peak-min-distance", type=int, default=3, help="Peak min-distance for center-unet evaluation.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
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

    predictions = run_prediction_table(
        records=records,
        output_dir=args.output_dir,
        checkpoint_path=args.model_checkpoint,
        method=args.method,
        device=args.device,
        mc_samples=args.mc_samples,
        save_overlays=True,
    )
    metrics = evaluate_predictions(predictions)
    save_json(Path(args.output_dir) / "metrics.json", metrics)
    print(f"Saved evaluation outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
