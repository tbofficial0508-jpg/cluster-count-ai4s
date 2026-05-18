from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor

from .common import clamp, ensure_dir, format_warnings, save_json, read_image
from .data import load_dataset_records
from .evaluate import summarise_grouped_metrics
from .features import build_feature_frame
from .metrics import compute_count_metrics
from .visualization import estimate_blur_score, save_overlay
from .watershed_baseline import run_watershed


def compute_tree_uncertainty(model: ExtraTreesRegressor, features: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    feature_values = np.asarray(features, dtype=np.float32)
    tree_predictions = np.stack([tree.predict(feature_values) for tree in model.estimators_], axis=0)
    return tree_predictions.mean(axis=0), tree_predictions.std(axis=0)


def evaluate_feature_ensemble(
    records: pd.DataFrame,
    model_bundle_path: str | Path,
    output_dir: str | Path,
) -> tuple[pd.DataFrame, dict[str, object]]:
    output_base = ensure_dir(output_dir)
    overlay_dir = ensure_dir(output_base / "overlays")
    bundle = joblib.load(model_bundle_path)
    model: ExtraTreesRegressor = bundle["model"]
    feature_columns: list[str] = bundle["feature_columns"]

    feature_frame = build_feature_frame(records)
    predictions, uncertainty = compute_tree_uncertainty(model, feature_frame[feature_columns])

    rows: list[dict[str, object]] = []
    for index, row in feature_frame.iterrows():
        image = read_image(row["image_path"])
        baseline = run_watershed(image)
        pred_count = float(predictions[index])
        count_std = float(uncertainty[index])
        blur_score = float(estimate_blur_score(image))
        baseline_count = float(baseline["count"])
        disagreement = abs(pred_count - baseline_count)
        disagreement_ratio = disagreement / max(pred_count, baseline_count, 1.0)
        confidence = clamp(1.0 - (0.35 * (count_std / max(pred_count, 1.0)) + 0.35 * disagreement_ratio + 0.30 * blur_score))

        warnings: list[str] = []
        if blur_score > 0.50:
            warnings.append("strong blur or low focus")
        if disagreement_ratio > 0.20:
            warnings.append("ensemble-baseline disagreement")
        if confidence < 0.55:
            warnings.append("low confidence")

        overlay_path = overlay_dir / f"{row['image_name']}_overlay.png"
        annotations = [
            f"pred={pred_count:.1f} baseline={baseline_count:.1f} conf={confidence:.2f}",
            f"uncertainty={count_std:.2f} blur={blur_score:.2f}",
            f"warnings={format_warnings(warnings) or 'none'}",
        ]
        if "true_count" in row and not pd.isna(row["true_count"]):
            annotations.insert(1, f"truth={float(row['true_count']):.1f}")
        save_overlay(image, overlay_path, points=np.asarray(baseline["points"]), title=str(row["image_name"]), annotations=annotations)

        result = row.to_dict()
        truth = float(row["true_count"])
        result.update(
            {
                "pred_count": pred_count,
                "baseline_count": baseline_count,
                "count_std": count_std,
                "blur_score_est": blur_score,
                "disagreement": disagreement,
                "disagreement_ratio": disagreement_ratio,
                "confidence": confidence,
                "warnings": format_warnings(warnings),
                "overlay_path": str(overlay_path),
                "abs_error": abs(pred_count - truth),
                "baseline_abs_error": abs(baseline_count - truth),
            }
        )
        rows.append(result)

    prediction_frame = pd.DataFrame(rows)
    prediction_frame.to_csv(output_base / "count.csv", index=False)

    metrics: dict[str, object] = {
        "n_images": int(len(prediction_frame)),
        "model": compute_count_metrics(prediction_frame["true_count"], prediction_frame["pred_count"]),
        "watershed": compute_count_metrics(prediction_frame["true_count"], prediction_frame["baseline_count"]),
        "hardest_examples": prediction_frame.sort_values("abs_error", ascending=False)
        .head(5)[["image_name", "true_count", "pred_count", "baseline_count", "abs_error", "confidence", "warnings", "overlay_path"]]
        .to_dict(orient="records"),
    }

    if "dataset" in prediction_frame.columns and len(prediction_frame["dataset"].unique()) == 1:
        dataset_name = str(prediction_frame["dataset"].iloc[0])
        if dataset_name == "synthetic":
            if {"blur_sigma", "cluster_strength"}.issubset(prediction_frame.columns):
                blur_bins = pd.cut(
                    prediction_frame["blur_sigma"],
                    bins=[0.0, 0.9, 1.4, 10.0],
                    labels=["low", "medium", "high"],
                    include_lowest=True,
                )
                cluster_bins = pd.cut(
                    prediction_frame["cluster_strength"],
                    bins=[0.0, 0.30, 0.55, 1.0],
                    labels=["low", "medium", "high"],
                    include_lowest=True,
                )
                synthetic_frame = prediction_frame.assign(blur_bucket=blur_bins, cluster_bucket=cluster_bins)
                metrics["failure_regimes"] = {
                    "blur_bucket": summarise_grouped_metrics(synthetic_frame, "blur_bucket", "pred_count"),
                    "cluster_bucket": summarise_grouped_metrics(synthetic_frame, "cluster_bucket", "pred_count"),
                }
                operating = prediction_frame[
                    (prediction_frame["blur_sigma"] <= 0.9) & (prediction_frame["cluster_strength"] <= 0.30)
                ].reset_index(drop=True)
                if len(operating) > 0:
                    metrics["operating_envelope"] = {
                        "description": "Synthetic operating envelope with blur_sigma <= 0.9 and cluster_strength <= 0.30.",
                        "n_images": int(len(operating)),
                        **compute_count_metrics(operating["true_count"], operating["pred_count"]),
                    }
        elif dataset_name == "s_bsst265":
            metrics["failure_regimes"] = {
                "snr_class": summarise_grouped_metrics(prediction_frame, "snr_class", "pred_count"),
                "test_class": summarise_grouped_metrics(prediction_frame, "test_class", "pred_count"),
                "magnification": summarise_grouped_metrics(prediction_frame, "magnification", "pred_count"),
                "modality": summarise_grouped_metrics(prediction_frame, "modality", "pred_count"),
                "preparation": summarise_grouped_metrics(prediction_frame, "preparation", "pred_count"),
                "diagnosis": summarise_grouped_metrics(prediction_frame, "diagnosis", "pred_count"),
            }

    save_json(output_base / "metrics.json", metrics)
    return prediction_frame, metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a feature-ensemble count model.")
    parser.add_argument("--manifest", type=Path, help="Manifest CSV for synthetic or custom data.")
    parser.add_argument("--dataset", type=str, help="Named dataset adapter: s_bsst265, bbbc004, or bbbc005.")
    parser.add_argument("--dataset-root", type=Path, help="Dataset root for named dataset adapters.")
    parser.add_argument("--model-bundle", type=Path, required=True, help="Path to feature_ensemble.joblib.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for count.csv, overlays, and metrics.")
    parser.add_argument("--split", type=str, help="Optional split filter.")
    parser.add_argument("--limit", type=int, default=0, help="Optional record limit.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    records = load_dataset_records(manifest_path=args.manifest, dataset_name=args.dataset, dataset_root=args.dataset_root)
    if args.split:
        records = records[records["split"] == args.split].reset_index(drop=True)
    if args.limit > 0:
        records = records.head(args.limit).reset_index(drop=True)
    evaluate_feature_ensemble(records, model_bundle_path=args.model_bundle, output_dir=args.output_dir)
    print(f"Saved feature-ensemble evaluation outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
