# Validation Report

Generated: 2026-05-18 05:08 UTC

## Summary

- Images evaluated: 3
- Density-model MAE: 28.37
- Watershed baseline MAE: 36.67

## Method Comparison

| Method | MAE | RMSE | MAPE | Bias |
| --- | ---: | ---: | ---: | ---: |
| Density / ensemble | 28.37 | 35.17 | 0.436 | -28.37 |
| Watershed baseline | 36.67 | 42.01 | 0.627 | -36.67 |

## Failure Regimes

These slices are where a non-expert user should expect the most count risk or the strongest need for overlay review.

### Blur Bucket

| Group | MAE | RMSE | MAPE | Bias |
| --- | ---: | ---: | ---: | ---: |
| medium | 15.40 | 19.51 | 0.310 | -15.40 |
| high | 54.31 | 54.31 | 0.687 | -54.31 |

### Cluster Bucket

| Group | MAE | RMSE | MAPE | Bias |
| --- | ---: | ---: | ---: | ---: |
| low | 28.87 | 38.48 | 0.405 | -28.87 |
| medium | 27.38 | 27.38 | 0.498 | -27.38 |

## Hardest Examples

| Image | Truth | Pred | Baseline | Abs Error | Confidence | Warnings | Overlay |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| synthetic_0022 | 79.0 | 24.7 | 16.0 | 54.3 | 0.69 | high clustering; strong blur or low focus; model-baseline disagreement | synthetic_0022_overlay.png |
| synthetic_0021 | 55.0 | 27.6 | 21.0 | 27.4 | 0.73 | high clustering; strong blur or low focus | synthetic_0021_overlay.png |
| synthetic_0023 | 28.0 | 24.6 | 15.0 | 3.4 | 0.75 | high clustering; model-baseline disagreement | synthetic_0023_overlay.png |

## Limitations

- This is a prototype benchmark and validation framework for robust cell counting under clustering and blur, not a clinically validated system.
- Confidence is heuristic and should be interpreted together with overlays, not as a standalone quality certificate.
- 0 image(s) fell below the default confidence threshold of 0.55 in this run.
