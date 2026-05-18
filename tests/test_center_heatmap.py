import math

import numpy as np

from cluster_count.center_heatmap import (
    build_center_heatmap,
    count_heatmap_peaks,
    inverse_sigmoid,
    summarize_peak_grid,
)


def test_build_center_heatmap_has_unit_peaks() -> None:
    points = np.asarray([[8.0, 8.0], [20.0, 22.0]], dtype=np.float32)
    heatmap = build_center_heatmap(points, shape=(32, 32), sigma=2.0)
    assert heatmap.shape == (32, 32)
    assert np.isclose(float(heatmap.max()), 1.0, atol=1e-6)
    assert heatmap[8, 8] > 0.99


def test_count_heatmap_peaks_recovers_two_centers() -> None:
    points = np.asarray([[10.0, 10.0], [24.0, 25.0]], dtype=np.float32)
    heatmap = build_center_heatmap(points, shape=(40, 40), sigma=1.8)
    peaks = count_heatmap_peaks(heatmap, threshold_abs=0.30, min_distance=3)
    assert len(peaks) == 2


def test_summarize_peak_grid_selects_best_parameters() -> None:
    heatmap_a = build_center_heatmap(np.asarray([[8.0, 8.0], [20.0, 20.0]], dtype=np.float32), shape=(32, 32), sigma=2.0)
    heatmap_b = build_center_heatmap(np.asarray([[12.0, 12.0]], dtype=np.float32), shape=(32, 32), sigma=2.0)
    summary = summarize_peak_grid(
        prediction_maps=[heatmap_a, heatmap_b],
        true_counts=[2.0, 1.0],
        thresholds=[0.25, 0.35, 0.45],
        min_distances=[2, 3],
    )
    assert summary["metrics"]["mape"] == 0.0


def test_inverse_sigmoid_round_trip() -> None:
    probability = 0.17
    recovered = 1.0 / (1.0 + math.exp(-inverse_sigmoid(probability)))
    assert np.isclose(recovered, probability, atol=1e-6)
