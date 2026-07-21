"""Train a simple CNN for Moderate-vs-Over diagnostic experiments."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from common import (
    BinaryImageDataset,
    add_per_time_accuracy,
    build_torchvision_binary_model,
    compute_binary_metrics,
    ensure_dir,
    patch_metadata_path,
    resolve_device,
    save_confusion_matrix_png,
    save_history_plot,
    save_json,
    set_seed,
    source_metadata_path,
    source_splits_path,
)


def load_training_frame(run_dir: Path, dataset_kind: str) -> pd.DataFrame:
    splits = pd.read_csv(source_splits_path(run_dir), dtype={"time_code": str})
    split_columns = splits[["source_image_id", "split"]].copy()
    if dataset_kind == "original":
        frame = pd.read_csv(source_metadata_path(run_dir), dtype={"time_code": str})
        frame = frame.merge(split_columns, on="source_image_id", how="inner")
        frame["image_path"] = frame["original_path"]
    elif dataset_kind == "patch":
        frame = pd.read_csv(patch_metadata_path(run_dir), dtype={"time_code": str})
        frame = frame.merge(split_columns, on="source_image_id", how="inner")
        frame["image_path"] = frame["patch_path"]
    else:
        raise ValueError(f"Unknown dataset kind: {dataset_kind}")
    frame["label"] = frame["label"].astype(int)
    return frame.sort_values(["split", "time_code", "source_image_id"]).reset_index(drop=True)


def make_loader(frame: pd.DataFrame, split: str, args: argparse.Namespace, shuffle: bool) -> DataLoader:
    split_frame = frame[frame["split"] == split].reset_index(drop=True)
    dataset = BinaryImageDataset(
        split_frame,
        image_column="image_path",
        image_size=args.image_size,
        variant=args.variant,
        augment=shuffle,
    )
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    return DataLoader(
        dataset,
        batch_size=args.batch_size if split == "train" else args.eval_batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        generator=generator,
    )


def run_epoch(model, loader, criterion, optimizer, device, train: bool) -> Tuple[float, float]:
    model.train(train)
    total_loss = 0.0
    total_correct = 0
    total_count = 0
    for images, labels, _ in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).long()
        if train:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(train):
            logits = model(images)
            loss = criterion(logits, labels)
            if train:
                loss.backward()
                optimizer.step()
        total_loss += float(loss.item()) * labels.numel()
        total_correct += int((logits.argmax(dim=1) == labels).sum().item())
        total_count += int(labels.numel())
    return total_loss / max(1, total_count), total_correct / max(1, total_count)


def predict(model, loader, device) -> pd.DataFrame:
    model.eval()
    rows = []
    split_frame = loader.dataset.dataframe
    with torch.no_grad():
        for images, _, indices in loader:
            images = images.to(device, non_blocking=True)
            probabilities = torch.softmax(model(images), dim=1).cpu()
            predictions = probabilities.argmax(dim=1)
            for batch_pos, local_index in enumerate(indices.tolist()):
                row = split_frame.iloc[int(local_index)].to_dict()
                row["pred_label"] = int(predictions[batch_pos].item())
                row["prob_moderate"] = float(probabilities[batch_pos, 0].item())
                row["prob_over"] = float(probabilities[batch_pos, 1].item())
                rows.append(row)
    return pd.DataFrame(rows)


def train(args: argparse.Namespace) -> Path:
    set_seed(args.seed)
    run_dir = Path(args.run_dir)
    out_dir = ensure_dir(run_dir / f"cnn_{args.dataset}_{args.variant}")
    frame = load_training_frame(run_dir, args.dataset)
    frame.to_csv(out_dir / "training_frame.csv", index=False, encoding="utf-8-sig")

    loaders = {
        "train": make_loader(frame, "train", args, shuffle=True),
        "val": make_loader(frame, "val", args, shuffle=False),
        "test": make_loader(frame, "test", args, shuffle=False),
    }
    device = resolve_device(args.device)
    model = build_torchvision_binary_model(args.model, num_classes=2, pretrained=args.pretrained).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    history: List[Dict[str, float]] = []
    best_val_acc = -1.0
    best_epoch = -1
    bad_epochs = 0
    started = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(model, loaders["train"], criterion, optimizer, device, train=True)
        val_loss, val_acc = run_epoch(model, loaders["val"], criterion, optimizer, device, train=False)
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
        })
        print(f"[{args.dataset}/{args.variant}] epoch {epoch:03d}: train_acc={train_acc:.4f} val_acc={val_acc:.4f}")
        if val_acc > best_val_acc + 1e-8:
            best_val_acc = val_acc
            best_epoch = epoch
            bad_epochs = 0
            torch.save({"model_state_dict": model.state_dict(), "args": vars(args), "epoch": epoch}, out_dir / "best_model.pth")
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                break

    checkpoint = torch.load(out_dir / "best_model.pth", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False, encoding="utf-8-sig")
    save_history_plot(history, out_dir / "training_curves.png")

    metrics = {
        "dataset": args.dataset,
        "variant": args.variant,
        "model": args.model,
        "best_epoch": best_epoch,
        "best_val_acc": best_val_acc,
        "training_time_seconds": time.perf_counter() - started,
    }
    for split, loader in loaders.items():
        predictions = predict(model, loader, device)
        predictions.to_csv(out_dir / f"{split}_predictions.csv", index=False, encoding="utf-8-sig")
        split_metrics = compute_binary_metrics(predictions["label"].astype(int), predictions["pred_label"].astype(int))
        split_metrics["per_time_accuracy"] = add_per_time_accuracy(predictions)
        metrics[split] = split_metrics
        save_confusion_matrix_png(split_metrics["confusion_matrix"], out_dir / f"{split}_confusion_matrix.png", f"{split} confusion")
    save_json(out_dir / "metrics.json", metrics)
    print(f"cnn diagnostics saved: {out_dir}")
    return out_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Moderate-vs-Over binary CNN.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--dataset", choices=("original", "patch"), required=True)
    parser.add_argument("--variant", choices=("rgb", "gray", "blur"), default="rgb")
    parser.add_argument("--model", default="resnet18")
    parser.add_argument("--pretrained", default="auto")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
