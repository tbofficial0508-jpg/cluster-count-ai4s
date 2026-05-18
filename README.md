# ClusterCount-AI4S: Validation-first cell counting for dense microscopy images

This project benchmarks automated cell counting under blur and clustering using
synthetic microscopy data and Broad Bioimage Benchmark Collection datasets. It
combines a density-map CNN, a watershed baseline, visual count overlays, error
metrics, Docker packaging, and CI tests to make cell-counting predictions
auditable rather than black-box.

It is intentionally not positioned as "a new cell-counting invention." Tools
such as Cellpose, StarDist, and CellProfiler already exist. The value here is a
reproducible AI-for-science prototype that asks a more practical question:

> Can a non-expert lab user trust the count under clustering, blur, and domain shift?

## Why this is useful

- Focuses on biomedical image-analysis validation rather than leaderboard chasing.
- Compares a tiny density-map CNN against a classical watershed baseline.
- Produces overlays, per-image warnings, confidence heuristics, and batch reports.
- Runs on synthetic data out of the box and can benchmark local microscopy datasets.
- Includes tests, Docker packaging, and GitHub Actions CI.

## Current dataset support

- Synthetic microscopy generator with exact known counts.
- `S-BSST265` local fluorescence dataset adapter with exact counts derived from instance masks.
- Optional BBBC004 adapter for dense overlap stress tests.
- Optional BBBC005 adapter for blur/count variation when local copies are available.

Because this repo lives under the GSK folder, the `S-BSST265` adapter will
auto-discover `../S-BSST265/dataset` when present.

## Repo layout

```text
cluster-count-ai4s/
  README.md
  Dockerfile
  pyproject.toml
  .github/workflows/ci.yml
  src/cluster_count/
    calibration.py
    center_heatmap.py
    features.py
    generate_synthetic.py
    patchwise.py
    train_density_cnn.py
    train_feature_ensemble.py
    train_patchwise_cnn.py
    train_patchwise_calibrator.py
    train_center_heatmap_unet.py
    predict.py
    watershed_baseline.py
    evaluate.py
    evaluate_feature_ensemble.py
    make_figures.py
    report.py
  tests/
    test_count_parser.py
    test_metrics.py
    test_synthetic_generator.py
  examples/
    demo_output/
```

## Quickstart

Install:

```bash
python -m pip install -e .[dev]
```

Generate synthetic training data:

```bash
python -m cluster_count.generate_synthetic --output artifacts/synth_train --num-images 80 --image-size 128 --seed 7
```

Train the tiny density-map CNN:

```bash
python -m cluster_count.train_density_cnn --manifest artifacts/synth_train/manifest.csv --output-dir artifacts/model --epochs 8
```

Train the new patchwise S-BSST265 path with magnification-aware scale normalization:

```bash
python -m cluster_count.train_patchwise_cnn --dataset s_bsst265 --dataset-root ..\\S-BSST265\\dataset --output-dir artifacts/patchwise_s_bsst265 --epochs 30 --device auto --patch-size 192 --patch-stride 160 --reference-magnification 40 --input-normalization zscore --eval-val-images
```

Train the lightweight calibrator that currently gives the best external generalization on `S-BSST265`:

```bash
python -m cluster_count.train_patchwise_calibrator --dataset s_bsst265 --dataset-root ..\\S-BSST265\\dataset --patchwise-checkpoint artifacts/patchwise_s_bsst265/model.pt --output-dir artifacts/patchwise_calibrator --device auto --patch-size 192 --patch-stride 160 --reference-magnification 40 --input-normalization zscore
```

Train the direct-supervision center-heatmap U-Net benchmark path:

```bash
python -m cluster_count.train_center_heatmap_unet --dataset s_bsst265 --dataset-root ..\\S-BSST265\\dataset --output-dir artifacts/center_unet_s_bsst265 --epochs 20 --device auto --patch-size 160 --patch-stride 128 --reference-magnification 40 --input-normalization zscore --eval-val-images
```

Evaluate on the local `S-BSST265` benchmark:

```bash
python -m cluster_count.evaluate --dataset s_bsst265 --dataset-root ..\\S-BSST265\\dataset --model-checkpoint artifacts/model/model.pt --output-dir artifacts/eval_s_bsst265 --limit 12
python -m cluster_count.report --metrics-json artifacts/eval_s_bsst265/metrics.json --predictions-csv artifacts/eval_s_bsst265/count.csv --output artifacts/eval_s_bsst265/report.md
```

Evaluate the patchwise path with metadata-aware grouping:

```bash
python -m cluster_count.evaluate --dataset s_bsst265 --dataset-root ..\\S-BSST265\\dataset --model-checkpoint artifacts/patchwise_s_bsst265/model.pt --method patchwise-cnn --output-dir artifacts/eval_patchwise_s_bsst265 --split test --device auto --patch-size 192 --patch-stride 160 --reference-magnification 40 --input-normalization zscore
python -m cluster_count.report --metrics-json artifacts/eval_patchwise_s_bsst265/metrics.json --predictions-csv artifacts/eval_patchwise_s_bsst265/count.csv --output artifacts/eval_patchwise_s_bsst265/report.md
```

Evaluate the preferred external path, which calibrates the raw patchwise predictor against watershed disagreement and uncertainty:

```bash
python -m cluster_count.evaluate --dataset s_bsst265 --dataset-root ..\\S-BSST265\\dataset --model-checkpoint artifacts/patchwise_calibrator/patchwise_calibrator.joblib --method patchwise-calibrated --output-dir artifacts/eval_patchwise_calibrated_s_bsst265 --split test --device auto --patch-size 192 --patch-stride 160 --reference-magnification 40 --input-normalization zscore
python -m cluster_count.report --metrics-json artifacts/eval_patchwise_calibrated_s_bsst265/metrics.json --predictions-csv artifacts/eval_patchwise_calibrated_s_bsst265/count.csv --output artifacts/eval_patchwise_calibrated_s_bsst265/report.md
```

Evaluate the center-heatmap path with the same reporting pipeline:

```bash
python -m cluster_count.evaluate --dataset s_bsst265 --dataset-root ..\\S-BSST265\\dataset --model-checkpoint artifacts/center_unet_s_bsst265/model.pt --method center-unet --output-dir artifacts/eval_center_unet_s_bsst265 --split test --device auto --patch-size 160 --patch-stride 128 --reference-magnification 40 --input-normalization zscore
python -m cluster_count.report --metrics-json artifacts/eval_center_unet_s_bsst265/metrics.json --predictions-csv artifacts/eval_center_unet_s_bsst265/count.csv --output artifacts/eval_center_unet_s_bsst265/report.md
```

Run the calibrated operating-range benchmark with convergence and publication figures:

```bash
python -m cluster_count.generate_synthetic --preset operating_v2 --output examples/demo_output/operating_v2_benchmark --num-images 3000 --image-size 128 --seed 71
python -m cluster_count.train_feature_ensemble --manifest examples/demo_output/operating_v2_benchmark/manifest.csv --output-dir examples/demo_output/operating_v2_feature_model --seed 71 --n-estimators 1000
python -m cluster_count.evaluate_feature_ensemble --manifest examples/demo_output/operating_v2_benchmark/manifest.csv --model-bundle examples/demo_output/operating_v2_feature_model/feature_ensemble.joblib --output-dir examples/demo_output/operating_v2_eval --split test
python -m cluster_count.make_figures --convergence-csv examples/demo_output/operating_v2_feature_model/convergence.csv --predictions-csv examples/demo_output/operating_v2_eval/count.csv --metrics-json examples/demo_output/operating_v2_eval/metrics.json --output-dir examples/demo_output/publication_figures/operating_v2
```

Run a single-image prediction:

```bash
python -m cluster_count.predict --image ..\\S-BSST265\\dataset\\rawimages\\normal_30.tif --model-checkpoint artifacts/model/model.pt --output-dir artifacts/predict_single
```

## Outputs

Each batch prediction or evaluation run produces:

- `count.csv`: per-image counts, errors, uncertainty, disagreement, confidence, and warnings.
- `overlays/*.png`: auditable image overlays with counted peaks and failure hints.
- `metrics.json`: aggregate accuracy and failure-regime summaries.
- `report.md`: markdown summary of where the method works and where it breaks.

The figure pipeline additionally produces:

- `convergence.png`: model-convergence chart across the ClusterCountModel tree schedule.
- `parity_plot.png`: predicted-vs-true count figure.
- `regime_heatmap.png`: blur/clustering failure-regime figure.
- `method_comparison.png`: ClusterCountModel vs watershed comparison chart.

## Benchmark Snapshot

- Controlled synthetic operating-range benchmark:
  `examples/demo_output/operating_v2_eval/metrics.json` records `4.67%` MAPE on `300` held-out test images for `ClusterCountModel`, versus `31.73%` MAPE for watershed.
- Operating envelope slice:
  `blur_sigma <= 0.9` and `cluster_strength <= 0.30` reached `4.45%` MAPE on `63` held-out test images.
- Local external benchmark:
  `S-BSST265` remains a hard domain-shift stress test; the repo keeps this explicit instead of hiding it. The curated external example run in `examples/demo_output/s_bsst265_calibrated_eval/metrics.json` uses the calibrated patchwise model and reduced held-out external MAPE to `34.3%`, MAE to `55.6`, and median absolute error to `12.6`. The raw patchwise density CNN sits just behind it at `36.5%` MAPE, while watershed remains much worse on proportional error (`134.9%` MAPE) despite a lower RMSE on some ultra-dense scenes.
- Best external-method selection:
  the repo now treats `patchwise-calibrated` as the preferred `S-BSST265` method because it delivered the strongest balance of accuracy and generalizability among the implemented external paths. The direct-supervision center-heatmap U-Net is retained as a benchmark adapter, but its latest full run collapsed on the heterogeneous external split and is therefore not the recommended default.
- Experimental direct-supervision path:
  the center-heatmap U-Net is now implemented as a stronger local-supervision benchmark, but in its current form it collapses on the full heterogeneous `S-BSST265` split (`100%` MAPE in the latest full run). Keeping that negative result visible is useful: it shows that direct supervision alone does not guarantee robustness without a better objective or architecture.

Publication-style figures generated from the benchmark live in:

- [Operating-Range Figures](</C:/Users/willb/OneDrive - Cranfield University/Documents/Graduate Roles/GSK - AIML Engineer AI for Science/cluster-count-ai4s/examples/demo_output/publication_figures/operating_v2>)
- [S-BSST265 Figures](</C:/Users/willb/OneDrive - Cranfield University/Documents/Graduate Roles/GSK - AIML Engineer AI for Science/cluster-count-ai4s/examples/demo_output/publication_figures/s_bsst265>)

## Validation-first design

- `generate_synthetic.py` creates clustered, blurred fluorescence-like images with exact counts.
- `train_density_cnn.py` fits a small density-map CNN that predicts a count heatmap.
- `train_patchwise_cnn.py` fits a scale-normalized patchwise density CNN for large heterogeneous microscopy images.
- `calibration.py` and `train_patchwise_calibrator.py` fit a lightweight external-domain calibrator that combines raw patchwise count, watershed count, disagreement, and uncertainty into the strongest current `S-BSST265` path.
- `train_center_heatmap_unet.py` fits an experimental center-heatmap U-Net directly on `S-BSST265` masks and tunes peak-extraction parameters on a held-out validation slice.
- The patchwise trainer now uses output-bias initialization, per-patch z-score normalization, and relative/log-count losses to avoid the zero-count collapse that naive density regression can show under domain shift.
- `center_heatmap.py` provides Gaussian center targets, a compact U-Net-style detector, stitched patchwise center inference, and validation-time peak-threshold tuning.
- `features.py` extracts physically interpretable image statistics used by the calibrated benchmark model.
- `train_feature_ensemble.py` fits `ClusterCountModel`, a tree-ensemble count calibrator, and exports convergence data plus feature importance.
- `watershed_baseline.py` provides a classical reference method.
- `predict.py` estimates counts, MC-dropout uncertainty, model-baseline disagreement, blur score, and clustering score.
- `evaluate.py` slices errors by blur/clustering buckets for synthetic data and by SNR, test class, magnification, modality, preparation, and diagnosis for `S-BSST265`.
- `patchwise.py` handles magnification parsing, scale normalization, patch extraction, and stitched patchwise inference.
- `evaluate_feature_ensemble.py` evaluates `ClusterCountModel` and emits overlays, uncertainty, and benchmark metrics.
- `make_figures.py` creates publication-quality convergence, parity, regime, and comparison figures for `ClusterCountModel`.
- `report.py` turns raw outputs into a recruiter-safe markdown validation report.

## What this repo does not claim

- It does not claim to solve cell counting in general.
- It is not clinically validated.
- It does not claim strong performance on organoids, brightfield, 3D stacks, or arbitrary domains unless measured.
- Confidence is a heuristic for review prioritization, not a formal uncertainty guarantee.

## Future Work

- Add a stronger external benchmark stack built around instance or center supervision, such as `Cellpose`, `StarDist`, or a center-heatmap `U-Net`, and compare them with the same overlay, uncertainty, and grouped-failure reporting used here.
- Replace the current single external path with regime-specific experts or calibrators split by magnification, modality, preparation, and morphology family, then evaluate them with grouped validation rather than relying on one pooled model.
- Run a formal human-agreement study on a stratified `S-BSST265` slice, reporting inter-rater and intra-rater repeatability, bias, and non-inferiority so that model error can be compared against realistic human counting variability instead of an arbitrary single threshold.
- Expand targeted annotation on the hardest external regimes, especially `20x`, `LSM`, and `otherspecimen` images, using active-learning or hard-case mining rather than uniform labeling.
- Freeze a final untouched external test split and move model selection onto a separate development split so that future gains remain statistically credible.
- Add a future session-specific calibration extension: a lightweight API or UI where a user labels a small number of images from one experimental batch, fits a temporary calibrator on top of a frozen base model, validates it on a mini holdout, and resets it cleanly when the assay or cluster type changes.
- Extend the framework from count-only adaptation toward full count-plus-audit workflows, including saved session metadata, reproducible calibration reports, and optional export of overlays and corrected counts for laboratory records.

## Docker

Build:

```bash
docker build -t cluster-count-ai4s .
```

Run:

```bash
docker run --rm -it cluster-count-ai4s
```

## Tests

```bash
pytest -q
```

## CV bullet

Built ClusterCount-AI4S, a biomedical image-analysis repo for automated cell counting in dense/blurred microscopy images, combining a density-map CNN, watershed baseline, BBBC benchmark evaluation, visual overlays, Docker packaging, pytest tests, and GitHub Actions CI.
