from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .common import ensure_dir


plt.style.use("seaborn-v0_8-whitegrid")


def save_training_convergence(convergence_csv: str | Path, output_path: str | Path) -> Path:
    frame = pd.read_csv(convergence_csv)
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(frame["n_estimators"], frame["train_mae"], marker="o", label="Train MAE", color="#0b5fff")
    axes[0].plot(frame["n_estimators"], frame["val_mae"], marker="o", label="Val MAE", color="#ff6f00")
    axes[0].set_title("ClusterCountModel Convergence")
    axes[0].set_xlabel("Number of Trees")
    axes[0].set_ylabel("Count MAE")
    axes[0].legend(frameon=True)

    axes[1].plot(frame["n_estimators"], frame["train_mape"] * 100.0, marker="o", label="Train MAPE", color="#0b5fff")
    axes[1].plot(frame["n_estimators"], frame["val_mape"] * 100.0, marker="o", label="Val MAPE", color="#ff6f00")
    axes[1].axhline(5.0, linestyle="--", linewidth=1.2, color="#111111", label="5% target")
    axes[1].set_title("Relative Error Convergence")
    axes[1].set_xlabel("Number of Trees")
    axes[1].set_ylabel("MAPE (%)")
    axes[1].legend(frameon=True)

    figure.tight_layout()
    destination = Path(output_path)
    ensure_dir(destination.parent)
    figure.savefig(destination, dpi=220, bbox_inches="tight")
    plt.close(figure)
    return destination


def save_parity_plot(predictions_csv: str | Path, output_path: str | Path) -> Path:
    frame = pd.read_csv(predictions_csv)
    truth = frame["true_count"].to_numpy(dtype=float)
    pred = frame["pred_count"].to_numpy(dtype=float)
    max_value = float(max(truth.max(), pred.max())) * 1.05

    figure, axis = plt.subplots(figsize=(6.5, 6.5))
    scatter = axis.scatter(
        truth,
        pred,
        c=frame.get("blur_sigma", frame.get("blur_score_est", pd.Series(np.zeros(len(frame))))),
        cmap="viridis",
        s=42,
        alpha=0.85,
        edgecolors="white",
        linewidths=0.4,
    )
    axis.plot([0, max_value], [0, max_value], linestyle="--", color="#111111", linewidth=1.2)
    axis.set_xlim(0, max_value)
    axis.set_ylim(0, max_value)
    axis.set_xlabel("True count")
    axis.set_ylabel("Predicted count")
    axis.set_title("Predicted vs True Count")
    colorbar = figure.colorbar(scatter, ax=axis, fraction=0.046, pad=0.04)
    colorbar.set_label("Blur proxy")
    figure.tight_layout()

    destination = Path(output_path)
    ensure_dir(destination.parent)
    figure.savefig(destination, dpi=240, bbox_inches="tight")
    plt.close(figure)
    return destination


def save_regime_heatmap(predictions_csv: str | Path, output_path: str | Path) -> Path | None:
    frame = pd.read_csv(predictions_csv)
    if not {"blur_sigma", "cluster_strength"}.issubset(frame.columns):
        return None

    bins_blur = [0.0, 0.9, 1.4, 10.0]
    bins_cluster = [0.0, 0.30, 0.55, 1.0]
    labels = ["low", "medium", "high"]
    frame = frame.copy()
    frame["blur_bucket"] = pd.cut(frame["blur_sigma"], bins=bins_blur, labels=labels, include_lowest=True)
    frame["cluster_bucket"] = pd.cut(frame["cluster_strength"], bins=bins_cluster, labels=labels, include_lowest=True)
    frame["ape_pct"] = (frame["pred_count"] - frame["true_count"]).abs() / np.maximum(frame["true_count"], 1.0) * 100.0

    pivot = frame.pivot_table(index="blur_bucket", columns="cluster_bucket", values="ape_pct", aggfunc="mean")
    pivot = pivot.reindex(index=labels, columns=labels)

    figure, axis = plt.subplots(figsize=(6.8, 5.4))
    image = axis.imshow(pivot.to_numpy(dtype=float), cmap="magma_r")
    axis.set_xticks(range(len(labels)), labels=labels)
    axis.set_yticks(range(len(labels)), labels=labels)
    axis.set_xlabel("Clustering bucket")
    axis.set_ylabel("Blur bucket")
    axis.set_title("Mean Absolute Percentage Error by Regime")

    for y in range(len(labels)):
        for x in range(len(labels)):
            value = pivot.iloc[y, x]
            label = "NA" if pd.isna(value) else f"{value:.1f}%"
            axis.text(x, y, label, ha="center", va="center", color="white", fontsize=10, fontweight="bold")

    colorbar = figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    colorbar.set_label("MAPE (%)")
    figure.tight_layout()

    destination = Path(output_path)
    ensure_dir(destination.parent)
    figure.savefig(destination, dpi=240, bbox_inches="tight")
    plt.close(figure)
    return destination


def save_method_comparison(metrics_json: str | Path, output_path: str | Path) -> Path:
    metrics = json.loads(Path(metrics_json).read_text(encoding="utf-8"))
    labels = ["ClusterCountModel", "Watershed"]
    mape_values = [metrics["model"]["mape"] * 100.0, metrics["watershed"]["mape"] * 100.0]
    mae_values = [metrics["model"]["mae"], metrics["watershed"]["mae"]]

    figure, axes = plt.subplots(1, 2, figsize=(9.6, 4.6))
    axes[0].bar(labels, mape_values, color=["#0b5fff", "#7b7b7b"])
    axes[0].axhline(5.0, linestyle="--", linewidth=1.2, color="#111111")
    axes[0].set_ylabel("MAPE (%)")
    axes[0].set_title("Relative Error")

    axes[1].bar(labels, mae_values, color=["#0b5fff", "#7b7b7b"])
    axes[1].set_ylabel("MAE")
    axes[1].set_title("Absolute Count Error")
    for axis in axes:
        axis.tick_params(axis="x", rotation=12)
    figure.tight_layout()

    destination = Path(output_path)
    ensure_dir(destination.parent)
    figure.savefig(destination, dpi=240, bbox_inches="tight")
    plt.close(figure)
    return destination


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create publication-style benchmark figures.")
    parser.add_argument("--convergence-csv", type=Path, help="Path to convergence.csv from ClusterCountModel training.")
    parser.add_argument("--predictions-csv", type=Path, help="Path to count.csv from evaluation.")
    parser.add_argument("--metrics-json", type=Path, help="Path to metrics.json from evaluation.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for figure outputs.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_dir = ensure_dir(args.output_dir)
    if args.convergence_csv:
        save_training_convergence(args.convergence_csv, output_dir / "convergence.png")
    if args.predictions_csv:
        save_parity_plot(args.predictions_csv, output_dir / "parity_plot.png")
        save_regime_heatmap(args.predictions_csv, output_dir / "regime_heatmap.png")
    if args.metrics_json:
        save_method_comparison(args.metrics_json, output_dir / "method_comparison.png")
    print(f"Saved figures to {output_dir}")


if __name__ == "__main__":
    main()
