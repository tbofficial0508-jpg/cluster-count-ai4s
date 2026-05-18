from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .common import ensure_dir, save_json, set_seed
from .data import DensityDataset, load_dataset_records, split_records
from .modeling import DensityCountingCNN, density_to_count


def run_epoch(
    model: DensityCountingCNN,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total_count_mae = 0.0
    total_items = 0

    for images, targets in loader:
        images = images.to(device=device, dtype=torch.float32)
        targets = targets.to(device=device, dtype=torch.float32)
        predictions = model(images)

        density_loss = F.mse_loss(predictions, targets)
        count_loss = F.l1_loss(density_to_count(predictions), density_to_count(targets))
        loss = density_loss + 0.20 * count_loss

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
    parser = argparse.ArgumentParser(description="Train a tiny density-map CNN for cell counting.")
    parser.add_argument("--manifest", type=Path, required=True, help="Training manifest CSV.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for checkpoint and logs.")
    parser.add_argument("--epochs", type=int, default=8, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=4, help="Mini-batch size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed.")
    parser.add_argument("--device", type=str, default="auto", help="cpu, cuda, or auto.")
    return parser


def select_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def main() -> None:
    args = build_parser().parse_args()
    set_seed(args.seed)
    output_dir = ensure_dir(args.output_dir)
    device = select_device(args.device)

    records = load_dataset_records(manifest_path=args.manifest)
    train_records, val_records = split_records(records)

    train_loader = DataLoader(DensityDataset(train_records, augment=True), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(DensityDataset(val_records, augment=False), batch_size=1, shuffle=False)

    model = DensityCountingCNN()
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    history_rows: list[dict[str, float | int]] = []
    best_val_loss = float("inf")
    checkpoint_path = output_dir / "model.pt"

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, device=device, optimizer=optimizer)
        with torch.no_grad():
            val_metrics = run_epoch(model, val_loader, device=device, optimizer=None)

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
                    "model_kwargs": {"base_channels": 16, "dropout": 0.10},
                    "history": history_rows,
                    "manifest": str(args.manifest),
                    "seed": args.seed,
                },
                checkpoint_path,
            )

    pd.DataFrame(history_rows).to_csv(output_dir / "training_history.csv", index=False)
    save_json(
        output_dir / "metrics.json",
        {
            "best_val_loss": best_val_loss,
            "epochs": args.epochs,
            "device": str(device),
            "n_train": int(len(train_records)),
            "n_val": int(len(val_records)),
        },
    )
    print(f"Saved checkpoint to {checkpoint_path}")


if __name__ == "__main__":
    main()
