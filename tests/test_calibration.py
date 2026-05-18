import numpy as np
import pandas as pd

from cluster_count.calibration import (
    PATCHWISE_CALIBRATION_FEATURE_COLUMNS,
    build_patchwise_calibration_features,
    compute_calibrator_uncertainty,
    fit_patchwise_calibrator,
)


def test_build_patchwise_calibration_features_has_expected_columns() -> None:
    frame = build_patchwise_calibration_features(
        pred_count=12.0,
        baseline_count=10.0,
        count_std=1.5,
        disagreement=2.0,
        confidence=0.8,
    )
    assert list(frame.columns) == PATCHWISE_CALIBRATION_FEATURE_COLUMNS
    assert frame.iloc[0]["pred_count"] == 12.0


def test_fit_patchwise_calibrator_and_uncertainty_runs() -> None:
    data = pd.DataFrame(
        [
            {"pred_count": 10.0, "baseline_count": 12.0, "count_std": 1.0, "disagreement": 2.0, "confidence": 0.8, "true_count": 11.0},
            {"pred_count": 20.0, "baseline_count": 18.0, "count_std": 1.2, "disagreement": 2.0, "confidence": 0.7, "true_count": 19.0},
            {"pred_count": 30.0, "baseline_count": 26.0, "count_std": 2.0, "disagreement": 4.0, "confidence": 0.6, "true_count": 28.0},
            {"pred_count": 40.0, "baseline_count": 45.0, "count_std": 2.5, "disagreement": 5.0, "confidence": 0.5, "true_count": 43.0},
            {"pred_count": 55.0, "baseline_count": 50.0, "count_std": 3.0, "disagreement": 5.0, "confidence": 0.4, "true_count": 52.0},
        ]
    )
    fitted = fit_patchwise_calibrator(data, n_estimators=32, min_samples_leaf=1, max_depth=4, cv_folds=2)
    model = fitted["pipeline"].named_steps["reg"]
    features = data[PATCHWISE_CALIBRATION_FEATURE_COLUMNS].iloc[:2].reset_index(drop=True)
    mean, std = compute_calibrator_uncertainty(model, features)
    assert mean.shape == (2,)
    assert std.shape == (2,)
    assert np.all(mean > 0.0)
