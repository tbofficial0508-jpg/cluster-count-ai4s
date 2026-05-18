# Validation Report

Generated: 2026-05-18 12:55 UTC

## Summary

- Images evaluated: 300
- Density-model MAE: 1.39
- Watershed baseline MAE: 11.03

## Method Comparison

| Method | MAE | RMSE | MAPE | Bias |
| --- | ---: | ---: | ---: | ---: |
| Density / ensemble | 1.39 | 1.81 | 0.047 | -0.07 |
| Watershed baseline | 11.03 | 13.20 | 0.317 | -11.03 |

## Failure Regimes

These slices are where a non-expert user should expect the most count risk or the strongest need for overlay review.

### Blur Bucket

| Group | MAE | RMSE | MAPE | Bias |
| --- | ---: | ---: | ---: | ---: |
| low | 1.41 | 1.84 | 0.047 | -0.02 |
| medium | 1.38 | 1.77 | 0.046 | -0.12 |

### Cluster Bucket

| Group | MAE | RMSE | MAPE | Bias |
| --- | ---: | ---: | ---: | ---: |
| low | 1.36 | 1.75 | 0.043 | 0.05 |
| medium | 1.43 | 1.86 | 0.051 | -0.20 |

## Hardest Examples

| Image | Truth | Pred | Baseline | Abs Error | Confidence | Warnings | Overlay |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| synthetic_2756 | 52.0 | 46.3 | 26.0 | 5.7 | 0.57 | strong blur or low focus; ensemble-baseline disagreement | synthetic_2756_overlay.png |
| synthetic_2878 | 35.0 | 40.4 | 27.0 | 5.4 | 0.61 | strong blur or low focus; ensemble-baseline disagreement | synthetic_2878_overlay.png |
| synthetic_2930 | 38.0 | 32.9 | 22.0 | 5.1 | 0.60 | strong blur or low focus; ensemble-baseline disagreement | synthetic_2930_overlay.png |
| synthetic_2744 | 55.0 | 50.1 | 25.0 | 4.9 | 0.55 | strong blur or low focus; ensemble-baseline disagreement | synthetic_2744_overlay.png |
| synthetic_2951 | 21.0 | 16.2 | 10.0 | 4.8 | 0.57 | strong blur or low focus; ensemble-baseline disagreement | synthetic_2951_overlay.png |

## Limitations

- This is a prototype benchmark and validation framework for robust cell counting under clustering and blur, not a clinically validated system.
- Confidence is heuristic and should be interpreted together with overlays, not as a standalone quality certificate.
- 42 image(s) fell below the default confidence threshold of 0.55 in this run.
