import pytest

from cluster_count.metrics import absolute_count_error, compute_count_metrics, relative_count_error


def test_absolute_and_relative_error() -> None:
    assert absolute_count_error(10, 13) == 3.0
    assert relative_count_error(10, 13) == pytest.approx(0.3)


def test_compute_count_metrics() -> None:
    metrics = compute_count_metrics([10, 20, 30], [12, 18, 33])
    assert metrics["mae"] == pytest.approx((2 + 2 + 3) / 3)
    assert metrics["bias"] == pytest.approx((2 - 2 + 3) / 3)
    assert metrics["n"] == 3.0

