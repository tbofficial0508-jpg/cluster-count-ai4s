from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline


PATCHWISE_CALIBRATION_FEATURE_COLUMNS = [
    "pred_count",
    "baseline_count",
    "count_std",
    "disagreement",
    "confidence",
]


def build_patchwise_calibration_features(
    pred_count: float,
    baseline_count: float,
    count_std: float,
    disagreement: float,
    confidence: float,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "pred_count": float(pred_count),
                "baseline_count": float(baseline_count),
                "count_std": float(count_std),
                "disagreement": float(disagreement),
                "confidence": float(confidence),
            }
        ]
    )


def compute_calibrator_uncertainty(
    model: ExtraTreesRegressor,
    features: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(features, dtype=np.float32)
    tree_predictions = np.stack([np.expm1(tree.predict(values)) for tree in model.estimators_], axis=0)
    return tree_predictions.mean(axis=0), tree_predictions.std(axis=0)


def build_patchwise_calibrator_pipeline(
    n_estimators: int = 1200,
    min_samples_leaf: int = 2,
    max_depth: int | None = 6,
    random_state: int = 7,
) -> Pipeline:
    preprocessor = ColumnTransformer(
        [
            (
                "num",
                Pipeline([("imputer", SimpleImputer(strategy="median"))]),
                PATCHWISE_CALIBRATION_FEATURE_COLUMNS,
            )
        ]
    )
    regressor = ExtraTreesRegressor(
        n_estimators=n_estimators,
        min_samples_leaf=min_samples_leaf,
        max_depth=max_depth,
        random_state=random_state,
    )
    return Pipeline([("pre", preprocessor), ("reg", regressor)])


def fit_patchwise_calibrator(
    predictions: pd.DataFrame,
    n_estimators: int = 1200,
    min_samples_leaf: int = 2,
    max_depth: int | None = 6,
    random_state: int = 7,
    cv_folds: int = 5,
) -> dict[str, object]:
    required = set(PATCHWISE_CALIBRATION_FEATURE_COLUMNS + ["true_count"])
    missing = required.difference(predictions.columns)
    if missing:
        raise ValueError(f"Missing required columns for patchwise calibration: {sorted(missing)}")

    frame = predictions.reset_index(drop=True).copy()
    X = frame[PATCHWISE_CALIBRATION_FEATURE_COLUMNS]
    y = np.log1p(frame["true_count"].to_numpy(dtype=np.float64))

    pipeline = build_patchwise_calibrator_pipeline(
        n_estimators=n_estimators,
        min_samples_leaf=min_samples_leaf,
        max_depth=max_depth,
        random_state=random_state,
    )

    cv_mapes: list[float] = []
    if cv_folds >= 2 and len(frame) >= cv_folds:
        splitter = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        for train_index, val_index in splitter.split(frame):
            pipeline.fit(X.iloc[train_index], y[train_index])
            pred_val = np.expm1(pipeline.predict(X.iloc[val_index]))
            cv_mapes.append(float(mean_absolute_percentage_error(frame["true_count"].iloc[val_index], pred_val)))

    pipeline.fit(X, y)
    calibrated_train = np.expm1(pipeline.predict(X))

    return {
        "pipeline": pipeline,
        "feature_columns": list(PATCHWISE_CALIBRATION_FEATURE_COLUMNS),
        "cv_mape_mean": float(np.mean(cv_mapes)) if cv_mapes else None,
        "cv_mape_std": float(np.std(cv_mapes, ddof=0)) if cv_mapes else None,
        "train_mape": float(mean_absolute_percentage_error(frame["true_count"], calibrated_train)),
    }


def save_patchwise_calibrator_bundle(bundle: dict[str, object], path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, output_path)
    return output_path


def load_patchwise_calibrator_bundle(path: str | Path) -> dict[str, object]:
    return joblib.load(Path(path))
