from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor

from .common import ensure_dir, save_json, set_seed
from .data import load_dataset_records
from .features import build_feature_frame, export_feature_frame
from .metrics import compute_count_metrics


def build_model(n_estimators: int = 1000, seed: int = 7) -> ExtraTreesRegressor:
    return ExtraTreesRegressor(
        n_estimators=n_estimators,
        random_state=seed,
        min_samples_leaf=1,
        max_features="sqrt",
        n_jobs=1,
    )


def evaluate_frame(model: ExtraTreesRegressor, frame: pd.DataFrame, feature_columns: list[str]) -> dict[str, float]:
    predictions = model.predict(frame[feature_columns])
    return compute_count_metrics(frame["true_count"], predictions)


def fit_convergence_curve(
    train_frame: pd.DataFrame,
    val_frame: pd.DataFrame,
    feature_columns: list[str],
    seed: int,
) -> pd.DataFrame:
    schedule = [50, 100, 200, 400, 600, 800, 1000]
    rows: list[dict[str, float | int]] = []
    for n_estimators in schedule:
        model = build_model(n_estimators=n_estimators, seed=seed)
        model.fit(train_frame[feature_columns], train_frame["true_count"])
        train_metrics = evaluate_frame(model, train_frame, feature_columns)
        val_metrics = evaluate_frame(model, val_frame, feature_columns)
        rows.append(
            {
                "n_estimators": n_estimators,
                "train_mae": train_metrics["mae"],
                "train_mape": train_metrics["mape"],
                "val_mae": val_metrics["mae"],
                "val_mape": val_metrics["mape"],
            }
        )
    return pd.DataFrame(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a feature-ensemble count calibrator on microscopy images.")
    parser.add_argument("--manifest", type=Path, help="Manifest CSV for synthetic or custom data.")
    parser.add_argument("--dataset", type=str, help="Named dataset adapter: s_bsst265, bbbc004, or bbbc005.")
    parser.add_argument("--dataset-root", type=Path, help="Dataset root for named dataset adapters.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for feature model artifacts.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed.")
    parser.add_argument("--train-split", type=str, default="train", help="Training split label.")
    parser.add_argument("--val-split", type=str, default="val", help="Validation split label.")
    parser.add_argument("--n-estimators", type=int, default=1000, help="Number of trees in the final ensemble.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    set_seed(args.seed)
    output_dir = ensure_dir(args.output_dir)

    records = load_dataset_records(manifest_path=args.manifest, dataset_name=args.dataset, dataset_root=args.dataset_root)
    feature_frame = build_feature_frame(records)
    export_feature_frame(feature_frame, output_dir / "features.csv")

    train_frame = feature_frame[feature_frame["split"] == args.train_split].reset_index(drop=True)
    if len(train_frame) == 0:
        raise ValueError(f"No training rows found for split={args.train_split!r}.")

    if args.val_split in set(feature_frame["split"]):
        val_frame = feature_frame[feature_frame["split"] == args.val_split].reset_index(drop=True)
    else:
        cutoff = max(1, int(round(len(train_frame) * 0.8)))
        val_frame = train_frame.iloc[cutoff:].copy()
        train_frame = train_frame.iloc[:cutoff].copy()
    if len(val_frame) == 0:
        raise ValueError("Validation frame is empty; provide a larger dataset or an explicit val split.")

    feature_columns = [
        column
        for column in feature_frame.columns
        if column
        not in {
            "dataset",
            "image_name",
            "image_path",
            "mask_path",
            "points_path",
            "density_path",
            "split",
            "count",
            "true_count",
            "overlay_path",
            "blur_sigma",
            "cluster_strength",
        }
        and pd.api.types.is_numeric_dtype(feature_frame[column])
    ]
    convergence = fit_convergence_curve(train_frame, val_frame, feature_columns, seed=args.seed)
    convergence.to_csv(output_dir / "convergence.csv", index=False)

    model = build_model(n_estimators=args.n_estimators, seed=args.seed)
    model.fit(train_frame[feature_columns], train_frame["true_count"])
    joblib.dump(
        {
            "model": model,
            "feature_columns": feature_columns,
            "seed": args.seed,
            "train_split": args.train_split,
            "val_split": args.val_split,
        },
        output_dir / "feature_ensemble.joblib",
    )

    train_pred = model.predict(train_frame[feature_columns])
    val_pred = model.predict(val_frame[feature_columns])
    feature_importance = pd.DataFrame(
        {
            "feature": feature_columns,
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)
    feature_importance.to_csv(output_dir / "feature_importance.csv", index=False)

    save_json(
        output_dir / "metrics.json",
        {
            "train": compute_count_metrics(train_frame["true_count"], train_pred),
            "val": compute_count_metrics(val_frame["true_count"], val_pred),
            "n_train": int(len(train_frame)),
            "n_val": int(len(val_frame)),
            "n_estimators": args.n_estimators,
            "feature_columns": feature_columns,
        },
    )
    print(f"Saved feature ensemble artifacts to {output_dir}")


if __name__ == "__main__":
    main()
