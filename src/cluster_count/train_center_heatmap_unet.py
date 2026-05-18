from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .center_heatmap import (
    CenterHeatmapUNet,
    PatchCenterHeatmapDataset,
    estimate_initial_center_bias,
    load_center_checkpoint,
    predict_patchwise_center_heatmap_map,
    sigmoid_focal_loss,
    soft_dice_loss,
    summarize_peak_grid,
)
from .common import ensure_dir, read_image, save_json, select_device, set_seed
from .data import load_dataset_records
from .evaluate import evaluate_predictions
from .patchwise import split_training_records, standardize_patch_tensor
from .predict import run_prediction_table


def parse_float_grid(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def parse_int_grid(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def run_epoch(
    model: CenterHeatmapUNet,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    focal_loss_weight: float = 1.0,
    dice_loss_weight: float = 1.0,
    mass_loss_weight: float = 0.25,
    input_normalization: str = "zscore",
) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total_mass_mae = 0.0
    total_items = 0

    for images, targets in loader:
        images = images.to(device=device, dtype=torch.float32)
        targets = targets.to(device=device, dtype=torch.float32)
        if input_normalization == "zscore":
            images = standardize_patch_tensor(images)

        logits = model(images)
        probabilities = torch.sigmoid(logits)
        target_mass = targets.sum(dim=(-2, -1))
        pred_mass = probabilities.sum(dim=(-2, -1))

        focal_loss = sigmoid_focal_loss(logits, targets)
        dice_loss = soft_dice_loss(logits, targets)
        mass_loss = F.l1_loss(pred_mass, target_mass) / float(images.shape[-2] * images.shape[-1])
        loss = focal_loss_weight * focal_loss + dice_loss_weight * dice_loss + mass_loss_weight * mass_loss

        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        batch_size = images.shape[0]
        total_items += batch_size
        total_loss += float(loss.item()) * batch_size
        total_mass_mae += float(F.l1_loss(pred_mass, target_mass).item()) * batch_size

    return {
        "loss": total_loss / max(total_items, 1),
        "mass_mae": total_mass_mae / max(total_items, 1),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a center-heatmap U-Net on microscopy masks.")
    parser.add_argument("--manifest", type=Path, help="Optional manifest CSV.")
    parser.add_argument("--dataset", type=str, default="s_bsst265", help="Named dataset adapter.")
    parser.add_argument("--dataset-root", type=Path, help="Dataset root for named dataset adapters.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for checkpoint and logs.")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=8, help="Mini-batch size.")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed.")
    parser.add_argument("--device", type=str, default="auto", help="cpu, cuda, or auto.")
    parser.add_argument("--patch-size", type=int, default=192, help="Patch size in pixels after scale normalization.")
    parser.add_argument("--patch-stride", type=int, default=160, help="Patch stride in pixels after scale normalization.")
    parser.add_argument("--heatmap-sigma", type=float, default=2.2, help="Gaussian sigma for center heatmap targets.")
    parser.add_argument("--reference-magnification", type=float, default=40.0, help="Reference magnification for scale normalization.")
    parser.add_argument("--min-scale", type=float, default=0.50, help="Minimum allowed scale factor.")
    parser.add_argument("--max-scale", type=float, default=4.00, help="Maximum allowed scale factor.")
    parser.add_argument("--empty-patch-keep-prob", type=float, default=0.15, help="Probability of keeping empty patches.")
    parser.add_argument("--holdout-fraction", type=float, default=0.20, help="Validation holdout fraction when no val split exists.")
    parser.add_argument("--train-split", type=str, default="train", help="Training split label.")
    parser.add_argument("--val-split", type=str, default="val", help="Validation split label.")
    parser.add_argument("--limit-records", type=int, default=0, help="Optional record limit for quick experiments.")
    parser.add_argument("--max-train-patches", type=int, default=0, help="Optional cap on training patches.")
    parser.add_argument("--max-val-patches", type=int, default=0, help="Optional cap on validation patches.")
    parser.add_argument("--focal-loss-weight", type=float, default=1.0, help="Weight for sigmoid focal loss.")
    parser.add_argument("--dice-loss-weight", type=float, default=1.0, help="Weight for soft Dice loss.")
    parser.add_argument("--mass-loss-weight", type=float, default=0.25, help="Weight for heatmap mass regularization.")
    parser.add_argument("--dropout", type=float, default=0.10, help="Dropout rate in the U-Net blocks.")
    parser.add_argument("--base-channels", type=int, default=24, help="Base channel width for the U-Net.")
    parser.add_argument("--input-normalization", type=str, choices=("none", "zscore"), default="zscore", help="Patch intensity normalization mode.")
    parser.add_argument("--peak-threshold-grid", type=str, default="0.20,0.25,0.30,0.35,0.40,0.45", help="Comma-separated peak thresholds to scan on the validation set.")
    parser.add_argument("--peak-min-distance-grid", type=str, default="2,3,4,5", help="Comma-separated peak min-distance values to scan on the validation set.")
    parser.add_argument("--peak-eval-interval", type=int, default=1, help="Epoch interval for full-image validation peak evaluation.")
    parser.add_argument("--eval-val-images", action="store_true", help="Run full image inference on held-out validation images after training.")
    return parser


def tune_peak_parameters(
    model: CenterHeatmapUNet,
    records: pd.DataFrame,
    device: torch.device,
    patch_size: int,
    patch_stride: int,
    reference_magnification: float,
    min_scale: float,
    max_scale: float,
    batch_size: int,
    input_normalization: str,
    threshold_grid: list[float],
    min_distance_grid: list[int],
) -> dict[str, object]:
    prediction_maps: list[np.ndarray] = []
    true_counts: list[float] = []
    for _, row in records.iterrows():
        heatmap_result = predict_patchwise_center_heatmap_map(
            model=model,
            image=read_image(row["image_path"]),
            device=device,
            magnification=row.get("magnification"),
            reference_magnification=reference_magnification,
            min_scale=min_scale,
            max_scale=max_scale,
            patch_size=patch_size,
            stride=patch_stride,
            mc_samples=1,
            batch_size=batch_size,
            input_normalization=input_normalization,
            peak_threshold=threshold_grid[0],
            peak_min_distance=min_distance_grid[0],
        )
        prediction_maps.append(np.asarray(heatmap_result["heatmap"], dtype=np.float32))
        true_counts.append(float(row["true_count"]))
    return summarize_peak_grid(
        prediction_maps=prediction_maps,
        true_counts=true_counts,
        thresholds=threshold_grid,
        min_distances=min_distance_grid,
    )


def main() -> None:
    args = build_parser().parse_args()
    set_seed(args.seed)
    output_dir = ensure_dir(args.output_dir)
    device = select_device(args.device)

    records = load_dataset_records(manifest_path=args.manifest, dataset_name=args.dataset, dataset_root=args.dataset_root)
    if args.limit_records > 0:
        records = records.head(args.limit_records).reset_index(drop=True)

    train_records, val_records = split_training_records(
        records,
        train_split=args.train_split,
        val_split=args.val_split,
        holdout_fraction=args.holdout_fraction,
        seed=args.seed,
    )

    train_dataset = PatchCenterHeatmapDataset(
        train_records,
        patch_size=args.patch_size,
        stride=args.patch_stride,
        sigma=args.heatmap_sigma,
        reference_magnification=args.reference_magnification,
        min_scale=args.min_scale,
        max_scale=args.max_scale,
        empty_patch_keep_prob=args.empty_patch_keep_prob,
        augment=True,
        seed=args.seed,
        max_patches=args.max_train_patches,
    )
    val_dataset = PatchCenterHeatmapDataset(
        val_records,
        patch_size=args.patch_size,
        stride=args.patch_stride,
        sigma=args.heatmap_sigma,
        reference_magnification=args.reference_magnification,
        min_scale=args.min_scale,
        max_scale=args.max_scale,
        empty_patch_keep_prob=1.0,
        augment=False,
        seed=args.seed,
        max_patches=args.max_val_patches,
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    threshold_grid = parse_float_grid(args.peak_threshold_grid)
    min_distance_grid = parse_int_grid(args.peak_min_distance_grid)

    model = CenterHeatmapUNet(base_channels=args.base_channels, dropout=args.dropout)
    with torch.no_grad():
        model.head.bias.fill_(estimate_initial_center_bias(train_dataset))
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    history_rows: list[dict[str, float | int]] = []
    best_val_loss = float("inf")
    best_val_peak_mape = float("inf")
    checkpoint_path = output_dir / "model.pt"
    best_tuned: dict[str, object] | None = None

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model,
            train_loader,
            device=device,
            optimizer=optimizer,
            focal_loss_weight=args.focal_loss_weight,
            dice_loss_weight=args.dice_loss_weight,
            mass_loss_weight=args.mass_loss_weight,
            input_normalization=args.input_normalization,
        )
        with torch.no_grad():
            val_metrics = run_epoch(
                model,
                val_loader,
                device=device,
                optimizer=None,
                focal_loss_weight=args.focal_loss_weight,
                dice_loss_weight=args.dice_loss_weight,
                mass_loss_weight=args.mass_loss_weight,
                input_normalization=args.input_normalization,
            )

        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_mass_mae": train_metrics["mass_mae"],
            "val_loss": val_metrics["loss"],
            "val_mass_mae": val_metrics["mass_mae"],
        }
        tuned: dict[str, object] | None = None
        if epoch % max(args.peak_eval_interval, 1) == 0:
            tuned = tune_peak_parameters(
                model=model,
                records=val_records,
                device=device,
                patch_size=args.patch_size,
                patch_stride=args.patch_stride,
                reference_magnification=args.reference_magnification,
                min_scale=args.min_scale,
                max_scale=args.max_scale,
                batch_size=args.batch_size,
                input_normalization=args.input_normalization,
                threshold_grid=threshold_grid,
                min_distance_grid=min_distance_grid,
            )
            row["val_peak_mape"] = float(tuned["metrics"]["mape"])
            row["val_peak_mae"] = float(tuned["metrics"]["mae"])
            row["peak_threshold"] = float(tuned["peak_threshold"])
            row["peak_min_distance"] = int(tuned["peak_min_distance"])

        history_rows.append(row)

        if val_metrics["loss"] <= best_val_loss:
            best_val_loss = val_metrics["loss"]

        if tuned is not None:
            tuned_mape = float(tuned["metrics"]["mape"])
            if tuned_mape < best_val_peak_mape or (
                abs(tuned_mape - best_val_peak_mape) < 1e-9 and val_metrics["loss"] <= best_val_loss
            ):
                best_val_peak_mape = tuned_mape
                best_tuned = tuned
                torch.save(
                    {
                        "state_dict": model.state_dict(),
                        "model_kwargs": {"base_channels": args.base_channels, "dropout": args.dropout},
                        "history": history_rows,
                        "dataset": args.dataset,
                        "seed": args.seed,
                        "model_family": "center_unet",
                        "reference_magnification": args.reference_magnification,
                        "patch_size": args.patch_size,
                        "patch_stride": args.patch_stride,
                        "input_normalization": args.input_normalization,
                        "heatmap_sigma": args.heatmap_sigma,
                        "min_scale": args.min_scale,
                        "max_scale": args.max_scale,
                        "peak_threshold": float(tuned["peak_threshold"]),
                        "peak_min_distance": int(tuned["peak_min_distance"]),
                        "val_peak_metrics": tuned["metrics"],
                    },
                    checkpoint_path,
                )
        elif best_tuned is None and val_metrics["loss"] <= best_val_loss:
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "model_kwargs": {"base_channels": args.base_channels, "dropout": args.dropout},
                    "history": history_rows,
                    "dataset": args.dataset,
                    "seed": args.seed,
                    "model_family": "center_unet",
                    "reference_magnification": args.reference_magnification,
                    "patch_size": args.patch_size,
                    "patch_stride": args.patch_stride,
                    "input_normalization": args.input_normalization,
                    "heatmap_sigma": args.heatmap_sigma,
                    "min_scale": args.min_scale,
                    "max_scale": args.max_scale,
                },
                checkpoint_path,
            )

    pd.DataFrame(history_rows).to_csv(output_dir / "training_history.csv", index=False)

    best_model, checkpoint = load_center_checkpoint(checkpoint_path, device=device)
    tuned = best_tuned or tune_peak_parameters(
        model=best_model,
        records=val_records,
        device=device,
        patch_size=args.patch_size,
        patch_stride=args.patch_stride,
        reference_magnification=args.reference_magnification,
        min_scale=args.min_scale,
        max_scale=args.max_scale,
        batch_size=args.batch_size,
        input_normalization=args.input_normalization,
        threshold_grid=threshold_grid,
        min_distance_grid=min_distance_grid,
    )
    checkpoint["peak_threshold"] = float(tuned["peak_threshold"])
    checkpoint["peak_min_distance"] = int(tuned["peak_min_distance"])
    checkpoint["val_peak_metrics"] = tuned["metrics"]
    torch.save(checkpoint, checkpoint_path)

    metrics_payload: dict[str, object] = {
        "best_val_loss": best_val_loss,
        "epochs": args.epochs,
        "device": str(device),
        "n_train_images": int(len(train_records)),
        "n_val_images": int(len(val_records)),
        "n_train_patches": int(len(train_dataset)),
        "n_val_patches": int(len(val_dataset)),
        "patch_size": args.patch_size,
        "patch_stride": args.patch_stride,
        "reference_magnification": args.reference_magnification,
        "input_normalization": args.input_normalization,
        "heatmap_sigma": args.heatmap_sigma,
        "peak_threshold": float(tuned["peak_threshold"]),
        "peak_min_distance": int(tuned["peak_min_distance"]),
        "val_peak_metrics": tuned["metrics"],
        "focal_loss_weight": args.focal_loss_weight,
        "dice_loss_weight": args.dice_loss_weight,
        "mass_loss_weight": args.mass_loss_weight,
        "dropout": args.dropout,
        "base_channels": args.base_channels,
    }

    if args.eval_val_images:
        val_eval_records = val_records.copy()
        val_eval_records["patch_size"] = args.patch_size
        val_eval_records["patch_stride"] = args.patch_stride
        val_eval_records["reference_magnification"] = args.reference_magnification
        val_eval_records["min_scale"] = args.min_scale
        val_eval_records["max_scale"] = args.max_scale
        val_eval_records["batch_size"] = args.batch_size
        val_eval_records["input_normalization"] = args.input_normalization
        val_eval_records["peak_threshold"] = float(tuned["peak_threshold"])
        val_eval_records["peak_min_distance"] = int(tuned["peak_min_distance"])
        val_eval_dir = ensure_dir(output_dir / "val_eval")
        val_predictions = run_prediction_table(
            records=val_eval_records,
            output_dir=val_eval_dir,
            checkpoint_path=checkpoint_path,
            method="center-unet",
            device=device,
            mc_samples=1,
            save_overlays=False,
        )
        metrics_payload["val_image_metrics"] = evaluate_predictions(val_predictions)["model"]

    save_json(output_dir / "metrics.json", metrics_payload)
    print(f"Saved center-heatmap checkpoint to {checkpoint_path}")


if __name__ == "__main__":
    main()
