from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .common import ensure_dir, save_json, set_seed, select_device
from .data import load_dataset_records
from .evaluate import evaluate_predictions
from .modeling import DensityCountingCNN, density_to_count
from .patchwise import PatchDensityDataset, inverse_softplus, split_training_records, standardize_patch_tensor
from .predict import run_prediction_table


def run_epoch(
    model: DensityCountingCNN,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    density_loss_weight: float = 2.0,
    count_loss_weight: float = 0.20,
    log_count_loss_weight: float = 1.0,
    relative_count_loss_weight: float = 2.0,
    input_normalization: str = "zscore",
) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total_count_mae = 0.0
    total_items = 0

    for images, targets in loader:
        images = images.to(device=device, dtype=torch.float32)
        targets = targets.to(device=device, dtype=torch.float32)
        if input_normalization == "zscore":
            images = standardize_patch_tensor(images)
        predictions = model(images)
        pred_counts = density_to_count(predictions)
        target_counts = density_to_count(targets)

        density_loss = F.mse_loss(predictions, targets)
        count_loss = F.l1_loss(pred_counts, target_counts)
        log_count_loss = F.smooth_l1_loss(torch.log1p(pred_counts), torch.log1p(target_counts))
        relative_count_loss = torch.mean(torch.abs(pred_counts - target_counts) / torch.clamp(target_counts, min=1.0))
        loss = (
            density_loss_weight * density_loss
            + count_loss_weight * count_loss
            + log_count_loss_weight * log_count_loss
            + relative_count_loss_weight * relative_count_loss
        )

        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        batch_size = images.shape[0]
        total_items += batch_size
        total_loss += float(loss.item()) * batch_size
        total_count_mae += float(count_loss.item()) * batch_size

    return {
        "loss": total_loss / max(total_items, 1),
        "count_mae": total_count_mae / max(total_items, 1),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a patchwise density CNN with scale normalization.")
    parser.add_argument("--manifest", type=Path, help="Optional manifest CSV.")
    parser.add_argument("--dataset", type=str, default="s_bsst265", help="Named dataset adapter.")
    parser.add_argument("--dataset-root", type=Path, help="Dataset root for named dataset adapters.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for checkpoint and logs.")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=4, help="Mini-batch size.")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed.")
    parser.add_argument("--device", type=str, default="auto", help="cpu, cuda, or auto.")
    parser.add_argument("--patch-size", type=int, default=256, help="Patch size in pixels after scale normalization.")
    parser.add_argument("--patch-stride", type=int, default=224, help="Patch stride in pixels after scale normalization.")
    parser.add_argument("--sigma", type=float, default=2.0, help="Gaussian sigma for density targets.")
    parser.add_argument("--reference-magnification", type=float, default=40.0, help="Reference magnification for scale normalization.")
    parser.add_argument("--min-scale", type=float, default=0.50, help="Minimum allowed scale factor.")
    parser.add_argument("--max-scale", type=float, default=4.00, help="Maximum allowed scale factor.")
    parser.add_argument("--empty-patch-keep-prob", type=float, default=0.20, help="Probability of keeping empty patches.")
    parser.add_argument("--holdout-fraction", type=float, default=0.20, help="Validation holdout fraction when no val split exists.")
    parser.add_argument("--train-split", type=str, default="train", help="Training split label.")
    parser.add_argument("--val-split", type=str, default="val", help="Validation split label.")
    parser.add_argument("--limit-records", type=int, default=0, help="Optional record limit for quick experiments.")
    parser.add_argument("--max-train-patches", type=int, default=0, help="Optional cap on training patches.")
    parser.add_argument("--max-val-patches", type=int, default=0, help="Optional cap on validation patches.")
    parser.add_argument("--eval-val-images", action="store_true", help="Run full patchwise inference on held-out val images after training.")
    parser.add_argument("--density-loss-weight", type=float, default=2.0, help="Weight for density-map reconstruction loss.")
    parser.add_argument("--count-loss-weight", type=float, default=0.20, help="Weight for absolute count loss.")
    parser.add_argument("--log-count-loss-weight", type=float, default=1.0, help="Weight for log-count regression loss.")
    parser.add_argument("--relative-count-loss-weight", type=float, default=2.0, help="Weight for relative count error loss.")
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout rate in the CNN decoder/encoder blocks.")
    parser.add_argument("--base-channels", type=int, default=16, help="Base channel width for the density CNN.")
    parser.add_argument("--input-normalization", type=str, choices=("none", "zscore"), default="zscore", help="Patch intensity normalization mode.")
    return parser


def estimate_initial_head_bias(dataset: PatchDensityDataset, patch_size: int) -> float:
    patch_counts = [float(sample[1].sum()) for sample in dataset.samples]
    mean_count = float(sum(patch_counts) / max(len(patch_counts), 1))
    pixel_rate = mean_count / float(patch_size * patch_size)
    return inverse_softplus(pixel_rate)


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

    train_dataset = PatchDensityDataset(
        train_records,
        patch_size=args.patch_size,
        stride=args.patch_stride,
        sigma=args.sigma,
        reference_magnification=args.reference_magnification,
        min_scale=args.min_scale,
        max_scale=args.max_scale,
        empty_patch_keep_prob=args.empty_patch_keep_prob,
        augment=True,
        seed=args.seed,
        max_patches=args.max_train_patches,
    )
    val_dataset = PatchDensityDataset(
        val_records,
        patch_size=args.patch_size,
        stride=args.patch_stride,
        sigma=args.sigma,
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

    model = DensityCountingCNN(base_channels=args.base_channels, dropout=args.dropout)
    with torch.no_grad():
        model.head.bias.fill_(estimate_initial_head_bias(train_dataset, patch_size=args.patch_size))
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    history_rows: list[dict[str, float | int]] = []
    best_val_loss = float("inf")
    checkpoint_path = output_dir / "model.pt"

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model,
            train_loader,
            device=device,
            optimizer=optimizer,
            density_loss_weight=args.density_loss_weight,
            count_loss_weight=args.count_loss_weight,
            log_count_loss_weight=args.log_count_loss_weight,
            relative_count_loss_weight=args.relative_count_loss_weight,
            input_normalization=args.input_normalization,
        )
        with torch.no_grad():
            val_metrics = run_epoch(
                model,
                val_loader,
                device=device,
                optimizer=None,
                density_loss_weight=args.density_loss_weight,
                count_loss_weight=args.count_loss_weight,
                log_count_loss_weight=args.log_count_loss_weight,
                relative_count_loss_weight=args.relative_count_loss_weight,
                input_normalization=args.input_normalization,
            )

        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_count_mae": train_metrics["count_mae"],
            "val_loss": val_metrics["loss"],
            "val_count_mae": val_metrics["count_mae"],
        }
        history_rows.append(row)

        if val_metrics["loss"] <= best_val_loss:
            best_val_loss = val_metrics["loss"]
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "model_kwargs": {"base_channels": args.base_channels, "dropout": args.dropout},
                    "history": history_rows,
                    "dataset": args.dataset,
                    "seed": args.seed,
                    "patchwise": True,
                    "reference_magnification": args.reference_magnification,
                    "patch_size": args.patch_size,
                    "patch_stride": args.patch_stride,
                    "input_normalization": args.input_normalization,
                },
                checkpoint_path,
            )

    pd.DataFrame(history_rows).to_csv(output_dir / "training_history.csv", index=False)

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
        "density_loss_weight": args.density_loss_weight,
        "count_loss_weight": args.count_loss_weight,
        "log_count_loss_weight": args.log_count_loss_weight,
        "relative_count_loss_weight": args.relative_count_loss_weight,
        "dropout": args.dropout,
        "base_channels": args.base_channels,
    }

    if args.eval_val_images:
        val_eval_records = val_records.copy()
        val_eval_records["patch_size"] = args.patch_size
        val_eval_records["patch_stride"] = args.patch_stride
        val_eval_records["reference_magnification"] = args.reference_magnification
        val_eval_records["batch_size"] = args.batch_size
        val_eval_records["input_normalization"] = args.input_normalization
        val_eval_dir = ensure_dir(output_dir / "val_eval")
        val_predictions = run_prediction_table(
            records=val_eval_records,
            output_dir=val_eval_dir,
            checkpoint_path=checkpoint_path,
            method="patchwise-cnn",
            device=device,
            mc_samples=4,
            save_overlays=False,
        )
        metrics_payload["val_image_metrics"] = evaluate_predictions(val_predictions)["model"]

    save_json(output_dir / "metrics.json", metrics_payload)
    print(f"Saved patchwise checkpoint to {checkpoint_path}")


if __name__ == "__main__":
    main()
