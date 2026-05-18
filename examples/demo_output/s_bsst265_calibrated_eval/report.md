# Validation Report

Generated: 2026-05-18 18:30 UTC

## Summary

- Images evaluated: 37
- Density-model MAE: 55.61
- Watershed baseline MAE: 60.35

## Method Comparison

| Method | MAE | RMSE | MAPE | Bias |
| --- | ---: | ---: | ---: | ---: |
| Density / ensemble | 55.61 | 138.71 | 0.343 | -39.38 |
| Watershed baseline | 60.35 | 85.41 | 1.349 | 59.86 |

## Failure Regimes

These slices are where a non-expert user should expect the most count risk or the strongest need for overlay review.

### Snr Class

| Group | MAE | RMSE | MAPE | Bias |
| --- | ---: | ---: | ---: | ---: |
| <4 | 39.74 | 118.33 | 0.247 | -25.69 |
| >=4,<40 | 36.65 | 66.45 | 0.250 | -21.10 |
| >=40 | 167.70 | 292.16 | 0.950 | -142.63 |

### Test Class

| Group | MAE | RMSE | MAPE | Bias |
| --- | ---: | ---: | ---: | ---: |
| GNB-I | 87.29 | 124.01 | 0.212 | -54.38 |
| GNB-II | 30.54 | 37.69 | 0.435 | -30.54 |
| NB-I | 12.59 | 17.47 | 0.288 | 12.59 |
| NB-II | 69.16 | 96.73 | 0.290 | -67.63 |
| NB-III | 144.36 | 266.95 | 0.838 | -114.24 |
| NB-IV | 8.58 | 9.33 | 0.389 | 0.76 |
| NC-I | 2.35 | 3.34 | 0.077 | 0.57 |
| NC-II | 74.53 | 172.68 | 0.215 | -55.75 |
| NC-III | 2.77 | 3.31 | 0.135 | 1.83 |
| TS | 47.13 | 51.16 | 0.413 | -19.91 |

### Magnification

| Group | MAE | RMSE | MAPE | Bias |
| --- | ---: | ---: | ---: | ---: |
| 10x | 455.96 | 455.96 | 0.547 | -455.96 |
| 20x | 186.93 | 304.32 | 0.316 | -156.11 |
| 40x | 11.87 | 12.65 | 0.170 | 6.91 |
| 63x | 24.33 | 40.73 | 0.391 | -11.21 |

### Modality

| Group | MAE | RMSE | MAPE | Bias |
| --- | ---: | ---: | ---: | ---: |
| FM | 38.43 | 95.77 | 0.248 | -24.89 |
| LSM | 144.36 | 266.95 | 0.838 | -114.24 |

### Preparation

| Group | MAE | RMSE | MAPE | Bias |
| --- | ---: | ---: | ---: | ---: |
| BM cytospin | 12.59 | 17.47 | 0.288 | 12.59 |
| Tumor touch imprint | 8.58 | 9.33 | 0.389 | 0.76 |
| cellline cytospin | 76.90 | 180.98 | 0.375 | -60.41 |
| cellline grown | 2.77 | 3.31 | 0.135 | 1.83 |
| tissue section | 55.97 | 83.39 | 0.346 | -36.82 |

### Diagnosis

| Group | MAE | RMSE | MAPE | Bias |
| --- | ---: | ---: | ---: | ---: |
| Ganglioneuroblastoma | 58.92 | 91.65 | 0.324 | -42.46 |
| Neuroblastoma | 75.67 | 173.58 | 0.531 | -56.51 |
| Wilms | 27.23 | 27.23 | 0.495 | 27.23 |
| normal (HaCaT) | 36.12 | 117.98 | 0.153 | -25.46 |

## Hardest Examples

| Image | Truth | Pred | Baseline | Abs Error | Confidence | Warnings | Overlay |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| otherspecimen_6 | 969.0 | 323.9 | 1024.0 | 645.1 | 0.52 | strong blur or low focus; model-baseline disagreement; low confidence | otherspecimen_6_overlay.png |
| normal_34 | 833.0 | 377.0 | 824.0 | 456.0 | 0.66 | strong blur or low focus; model-baseline disagreement | normal_34_overlay.png |
| Ganglioneuroblastoma_7 | 585.0 | 376.0 | 614.0 | 209.0 | 0.67 | strong blur or low focus; model-baseline disagreement | Ganglioneuroblastoma_7_overlay.png |
| otherspecimen_1 | 265.0 | 128.2 | 340.0 | 136.8 | 0.08 | strong blur or low focus; model-baseline disagreement; low confidence | otherspecimen_1_overlay.png |
| otherspecimen_8 | 147.0 | 79.6 | 245.0 | 67.4 | 0.00 | strong blur or low focus; model-baseline disagreement; low confidence | otherspecimen_8_overlay.png |

## Limitations

- This is a prototype benchmark and validation framework for robust cell counting under clustering and blur, not a clinically validated system.
- Confidence is heuristic and should be interpreted together with overlays, not as a standalone quality certificate.
- 24 image(s) fell below the default confidence threshold of 0.55 in this run.
