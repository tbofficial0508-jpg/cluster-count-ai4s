import numpy as np
import torch

from cluster_count.patchwise import (
    build_rescaled_density,
    compute_patch_starts,
    inverse_softplus,
    iter_patch_boxes,
    magnification_scale,
    parse_magnification,
    standardize_patch_tensor,
)


def test_parse_magnification_and_scale() -> None:
    assert parse_magnification("63x") == 63.0
    assert parse_magnification("20 X objective") == 20.0
    assert parse_magnification(None) is None
    assert magnification_scale("20x", reference_magnification=40.0) == 2.0
    assert 0.5 <= magnification_scale("1000x", reference_magnification=40.0) <= 4.0


def test_build_rescaled_density_preserves_count() -> None:
    mask = np.zeros((16, 16), dtype=np.int32)
    mask[4:6, 4:6] = 1
    mask[10:12, 10:12] = 2
    density = build_rescaled_density(mask, scale=1.75, sigma=1.5)
    assert density.shape[0] >= mask.shape[0]
    assert density.shape[1] >= mask.shape[1]
    assert np.isclose(float(density.sum()), 2.0, atol=1e-4)


def test_patch_boxes_cover_image_extent() -> None:
    starts = compute_patch_starts(length=10, patch_size=4, stride=3)
    assert starts == [0, 3, 6]

    boxes = iter_patch_boxes((11, 9), patch_size=4, stride=3)
    coverage = np.zeros((11, 9), dtype=np.int32)
    for y0, x0 in boxes:
        coverage[y0 : y0 + 4, x0 : x0 + 4] += 1
    assert coverage.min() >= 1


def test_standardize_patch_tensor_normalizes_each_patch() -> None:
    batch = torch.tensor(
        [
            [[[1.0, 2.0], [3.0, 4.0]]],
            [[[10.0, 10.0], [10.0, 10.0]]],
        ],
        dtype=torch.float32,
    )
    normalized = standardize_patch_tensor(batch)
    means = normalized.mean(dim=(-2, -1))
    assert torch.allclose(means[0], torch.tensor([0.0]), atol=1e-6)
    assert torch.allclose(normalized[1], torch.zeros_like(normalized[1]), atol=1e-6)


def test_inverse_softplus_round_trip() -> None:
    value = 0.0125
    recovered = torch.nn.functional.softplus(torch.tensor(inverse_softplus(value))).item()
    assert np.isclose(recovered, value, atol=1e-6)
