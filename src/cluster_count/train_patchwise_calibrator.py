from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .calibration import fit_patchwise_calibrator, save_patchwise_calibrator_bundle
from .common import ensure_dir, save_json, select_device, set_seed
from .data import load_dataset_records
from .evaluate import evaluate_predictions
from .patchwise import split_training_records
from .predict import run_prediction_table


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a lightweight calibrator on top of patchwise density predictions.")
    parser.add_argument("--manifest", type=Path, help="Optional manifest CSV.")
    parser.add_argument("--dataset", type=str, default="s_bsst265", help="Named dataset adapter.")
    parser.add_argument("--dataset-root", type=Path, help="Dataset root for named dataset adapters.")
    parser.add_argument("--patchwise-checkpoint", type=Path, required=True, help="Trained patchwise density CNN checkpoint.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for the calibrator bundle and diagnostics.")
    parser.add_argument("--device", type=str, default="auto", help="cpu, cuda, or auto.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed.")
    parser.add_argument("--train-split", type=str, default="train", help="Training split label.")
    parser.add_argument("--val-split", type=str, default="val", help="Validation split label.")
    parser.add_argument("--holdout-fraction", type=float, default=0.20, help="Validation holdout fraction when no val split exists.")
    parser.add_argument("--patch-size", type=int, default=192, help="Patch size for the raw patchwise predictor.")
    parser.add_argument("--patch-stride", type=int, default=160, help="Patch stride for the raw patchwise predictor.")
    parser.add_argument("--reference-magnification", type=float, default=40.0, help="Reference magnification for the raw patchwise predictor.")
    parser.add_argument("--min-scale", type=float, default=0.50, help="Minimum scale factor for the raw patchwise predictor.")
    parser.add_argument("--max-scale", type=float, default=4.00, help="Maximum scale factor for the raw patchwise predictor.")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for the raw patchwise predictor.")
    parser.add_argument("--input-normalization", type=str, choices=["none", "zscore"], default="zscore", help="Patch intensity normalization mode for the raw predictor.")
    parser.add_argument("--n-estimators", type=int, default=1200, help="Number of trees in the calibrator.")
    parser.add_argument("--min-samples-leaf", type=int, default=2, help="Minimum samples per leaf in the calibrator.")
    parser.add_argument("--max-depth", type=int, default=6, help="Maximum tree depth in the calibrator. Use 0 for unlimited depth.")
    parser.add_argument("--cv-folds", type=int, default=5, help="Cross-validation folds for training diagnostics.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    set_seed(args.seed)
    device = select_device(args.device)
    output_dir = ensure_dir(args.output_dir)

    records = load_dataset_records(manifest_path=args.manifest, dataset_name=args.dataset, dataset_root=args.dataset_root)
    train_records, _ = split_training_records(
        records,
        train_split=args.train_split,
        val_split=args.val_split,
        holdout_fraction=args.holdout_fraction,
        seed=args.seed,
    )
    train_records = train_records.copy()
    train_records["patch_size"] = args.patch_size
    train_records["patch_stride"] = args.patch_stride
    train_records["reference_magnification"] = args.reference_magnification
    train_records["min_scale"] = args.min_scale
    train_records["max_scale"] = args.max_scale
    train_records["batch_size"] = args.batch_size
    train_records["input_normalization"] = args.input_normalization

    raw_prediction_dir = ensure_dir(output_dir / "raw_train_predictions")
    raw_predictions = run_prediction_table(
        records=train_records,
        output_dir=raw_prediction_dir,
        checkpoint_path=args.patchwise_checkpoint,
        method="patchwise-cnn",
        device=device,
        mc_samples=1,
        save_overlays=False,
    )

    calibrator = fit_patchwise_calibrator(
        raw_predictions,
        n_estimators=args.n_estimators,
        min_samples_leaf=args.min_samples_leaf,
        max_depth=None if args.max_depth <= 0 else args.max_depth,
        random_state=args.seed,
        cv_folds=args.cv_folds,
    )

    bundle = {
        "model": calibrator["pipeline"].named_steps["reg"],
        "feature_columns": calibrator["feature_columns"],
        "patchwise_checkpoint": str(Path(args.patchwise_checkpoint).resolve()),
        "patchwise_config": {
            "patch_size": args.patch_size,
            "patch_stride": args.patch_stride,
            "reference_magnification": args.reference_magnification,
            "min_scale": args.min_scale,
            "max_scale": args.max_scale,
            "batch_size": args.batch_size,
            "input_normalization": args.input_normalization,
        },
        "calibrator_config": {
            "n_estimators": args.n_estimators,
            "min_samples_leaf": args.min_samples_leaf,
            "max_depth": None if args.max_depth <= 0 else args.max_depth,
            "seed": args.seed,
        },
        "train_metrics": {
            "raw_patchwise": evaluate_predictions(raw_predictions)["model"],
            "train_mape": calibrator["train_mape"],
            "cv_mape_mean": calibrator["cv_mape_mean"],
            "cv_mape_std": calibrator["cv_mape_std"],
        },
    }
    bundle_path = save_patchwise_calibrator_bundle(bundle, output_dir / "patchwise_calibrator.joblib")

    feature_array = np.asarray(raw_predictions[calibrator["feature_columns"]], dtype=np.float32)
    tree_predictions = np.stack([np.expm1(tree.predict(feature_array)) for tree in bundle["model"].estimators_], axis=0)
    calibrated_counts = tree_predictions.mean(axis=0)
    diagnostics = raw_predictions.copy()
    diagnostics["raw_pred_count"] = diagnostics["pred_count"]
    diagnostics["pred_count"] = calibrated_counts
    diagnostics["calibrated_abs_error"] = np.abs(diagnostics["pred_count"] - diagnostics["true_count"])
    diagnostics.to_csv(output_dir / "calibration_train_predictions.csv", index=False)

    save_json(
        output_dir / "metrics.json",
        {
            "bundle_path": str(bundle_path),
            "n_train_images": int(len(train_records)),
            "raw_patchwise_metrics": evaluate_predictions(raw_predictions)["model"],
            "calibrated_train_metrics": evaluate_predictions(diagnostics)["model"],
            "cv_mape_mean": calibrator["cv_mape_mean"],
            "cv_mape_std": calibrator["cv_mape_std"],
        },
    )
    print(f"Saved patchwise calibrator bundle to {bundle_path}")


if __name__ == "__main__":
    main()
