import numpy as np

from cluster_count.features import extract_count_features


def test_extract_count_features_has_expected_keys() -> None:
    image = np.zeros((32, 32), dtype=np.float32)
    image[8:12, 8:12] = 0.8
    features = extract_count_features(image)

    assert "sum" in features
    assert "fg_0.10" in features
    assert "cc_0.10" in features
    assert "pk_s1.0_p96" in features
    assert features["sum"] > 0.0
