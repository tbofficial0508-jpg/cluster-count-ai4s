from pathlib import Path
import shutil
import uuid

import numpy as np

from cluster_count.generate_synthetic import generate_dataset


def test_synthetic_generator_creates_manifest_and_density_maps() -> None:
    output_dir = Path("artifacts") / f"test_synthetic_{uuid.uuid4().hex}"
    if output_dir.exists():
        shutil.rmtree(output_dir)

    frame = generate_dataset(output_dir, num_images=4, image_size=64, min_cells=6, max_cells=10, seed=11)

    assert len(frame) == 4
    assert (output_dir / "manifest.csv").exists()
    assert (output_dir / "preview_overlay.png").exists()

    first = frame.iloc[0]
    density = np.load(output_dir / first["density_path"])
    assert density.shape == (64, 64)
    assert abs(float(density.sum()) - float(first["count"])) < 1e-3
    assert 6 <= int(first["count"]) <= 10
