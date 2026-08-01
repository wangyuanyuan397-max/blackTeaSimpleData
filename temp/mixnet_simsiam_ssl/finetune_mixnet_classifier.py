"""Finetune a MixNet-S classifier from SimSiam backbone weights."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from common import (
    ImageFolderDataset,
    TimmClassifier,
    build_eval_transform,
    build_finetune_train_transform,
    build_optimizer,
    build_warmup_cosine_scheduler,
    classification_metrics,
    load_backbone_checkpoint,
    make_worker_init_fn,
    resolve_project_path,
    save_json,
    seed_everything,
    write_history_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Finetune MixNet-S on labeled BaSiC train/val/test splits."
    )
    parser.add_argument("--dataset-root", default="datasets_01234_BaSic")
    parser.add_argument("--model-name", default="mixnet_s")
    parser.add_argument("--image-size", type=int, default=408)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--optimizer", choices=["adamw", "adam", "sgd"], default="adamw")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--backbone-lr-mult", type=float, default=1.0)
    parser.add_argument("--head-lr-mult", type=float, default=1.0)
    parser.add_argument("--warmup-epochs", type=int, default=2)
    parser.add_argument("--min-lr-ratio", type=float, default=0.01)
    parser.add_argument("--drop-rate", type=float, default=0.0)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--train-transform-mode", choices=["crop", "resize", "resize_flip"], default="crop")
    parser.add_argument("--train-crop-min", type=float, default=0.85)
    parser.add_argument("--train-jitter-strength", type=float, default=0.08)
    parser.add_argument(
        "--selection-metric",
        choices=["val_acc", "val_qwk", "val_macro_f1", "val_loss"],
        default="val_acc",
        help="Metric used to save best_model.pth and trigger patience.",
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--max-steps-per-epoch", type=int, default=0)
    parser.add_argument("--imagenet-pretrained", action="store_true")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--ssl-checkpoint",
        default=(
            "temp/mixnet_simsiam_ssl/runs/simsiam_mixnet_s_basic408/"
            "mixnet_s_simsiam_backbone.pth"
        ),
        help="Use 'none' to train without SSL initialization.",
    )
    parser.add_argument(
        "--output-dir",
        default="temp/mixnet_simsiam_ssl/runs/finetune_mixnet_s_basic408",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run one train batch and one validation pass without writing checkpoints.",
    )
    return parser.parse_args()


def make_loader(
    dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    seed: int,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=bool(num_workers > 0),
        worker_init_fn=make_worker_init_fn(seed),
        generator=generator,
    )


def run_train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: GradScaler,
    use_amp: bool,
    max_steps: int,
) -> dict[str, Any]:
    model.train()
    total_loss = 0.0
    total_samples = 0
    all_true: list[int] = []
    all_pred: list[int] = []

    for step, (images, labels, _paths) in enumerate(loader, start=1):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with autocast(device_type=device.type, enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_size = int(labels.size(0))
        total_loss += float(loss.detach().item()) * batch_size
        total_samples += batch_size
        predictions = logits.detach().argmax(dim=1)
        all_true.extend(labels.detach().cpu().tolist())
        all_pred.extend(predictions.cpu().tolist())

        if max_steps > 0 and step >= max_steps:
            break

    metrics = classification_metrics(
        all_true,
        all_pred,
        num_classes=getattr(loader.dataset, "num_classes", len(set(all_true))),
        loss=total_loss / max(1, total_samples),
    )
    metrics["samples"] = total_samples
    return metrics


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    use_amp: bool,
    num_classes: int,
) -> dict[str, Any]:
    model.eval()
    total_loss = 0.0
    total_samples = 0
    all_true: list[int] = []
    all_pred: list[int] = []

    for images, labels, _paths in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with autocast(device_type=device.type, enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, labels)

        batch_size = int(labels.size(0))
        total_loss += float(loss.detach().item()) * batch_size
        total_samples += batch_size
        predictions = logits.argmax(dim=1)
        all_true.extend(labels.cpu().tolist())
        all_pred.extend(predictions.cpu().tolist())

    metrics = classification_metrics(
        all_true,
        all_pred,
        num_classes=num_classes,
        loss=total_loss / max(1, total_samples),
    )
    metrics["samples"] = total_samples
    return metrics


def save_model_checkpoint(
    path: Path,
    model: TimmClassifier,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    epoch: int,
    args: argparse.Namespace,
    classes: list[str],
    val_metrics: dict[str, Any],
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "model_name": args.model_name,
            "image_size": args.image_size,
            "classes": classes,
            "class_to_idx": {name: index for index, name in enumerate(classes)},
            "model_state_dict": model.state_dict(),
            "backbone_state_dict": model.backbone.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "args": vars(args),
            "val_metrics": val_metrics,
        },
        path,
    )


def should_use_ssl_checkpoint(value: str) -> bool:
    return str(value).strip().lower() not in {"", "none", "null", "false", "0"}


def build_finetune_param_groups(model: TimmClassifier, args: argparse.Namespace):
    """Build optional differential-learning-rate parameter groups."""
    backbone_lr = args.lr * float(args.backbone_lr_mult)
    head_lr = args.lr * float(args.head_lr_mult)
    return [
        {"params": model.backbone.parameters(), "lr": backbone_lr, "group_name": "backbone"},
        {"params": model.head.parameters(), "lr": head_lr, "group_name": "head"},
    ]


def selection_value(metric_name: str, val_metrics: dict[str, Any]) -> float:
    if metric_name == "val_acc":
        return float(val_metrics["accuracy"])
    if metric_name == "val_qwk":
        return float(val_metrics["qwk"])
    if metric_name == "val_macro_f1":
        return float(val_metrics["macro_f1"])
    if metric_name == "val_loss":
        return float(val_metrics["loss"])
    raise ValueError(f"Unsupported selection metric: {metric_name}")


def is_better_selection(metric_name: str, value: float, best_value: float) -> bool:
    if metric_name == "val_loss":
        return value < best_value
    return value > best_value


def initial_best_selection(metric_name: str) -> float:
    return float("inf") if metric_name == "val_loss" else float("-inf")


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    dataset_root = resolve_project_path(args.dataset_root)
    output_dir = resolve_project_path(args.output_dir)
    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
        save_json(vars(args), output_dir / "args.json")

    train_transform = build_finetune_train_transform(
        image_size=args.image_size,
        crop_min=args.train_crop_min,
        jitter_strength=args.train_jitter_strength,
        mode=args.train_transform_mode,
    )
    eval_transform = build_eval_transform(args.image_size)
    train_dataset = ImageFolderDataset(dataset_root / "train", transform=train_transform)
    class_to_idx = train_dataset.class_to_idx
    val_dataset = ImageFolderDataset(dataset_root / "val", transform=eval_transform, class_to_idx=class_to_idx)
    test_dataset = ImageFolderDataset(dataset_root / "test", transform=eval_transform, class_to_idx=class_to_idx)

    if args.max_samples > 0:
        train_dataset.samples = train_dataset.samples[: args.max_samples]
        train_dataset.targets = train_dataset.targets[: args.max_samples]
        val_dataset.samples = val_dataset.samples[: args.max_samples]
        val_dataset.targets = val_dataset.targets[: args.max_samples]
        test_dataset.samples = test_dataset.samples[: args.max_samples]
        test_dataset.targets = test_dataset.targets[: args.max_samples]

    for dataset in (train_dataset, val_dataset, test_dataset):
        dataset.num_classes = len(train_dataset.classes)

    train_loader = make_loader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        seed=args.seed,
    )
    val_loader = make_loader(
        val_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        seed=args.seed + 1000,
    )
    test_loader = make_loader(
        test_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        seed=args.seed + 2000,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TimmClassifier(
        model_name=args.model_name,
        num_classes=len(train_dataset.classes),
        pretrained=args.imagenet_pretrained,
        image_size=args.image_size,
        drop_rate=args.drop_rate,
    )

    ssl_load_info = None
    if should_use_ssl_checkpoint(args.ssl_checkpoint):
        ssl_path = resolve_project_path(args.ssl_checkpoint)
        if not ssl_path.is_file():
            raise FileNotFoundError(
                f"SSL checkpoint not found: {args.ssl_checkpoint}. Run train_simsiam_pretrain.py first "
                "or pass --ssl-checkpoint none."
            )
        missing, unexpected = load_backbone_checkpoint(model.backbone, ssl_path, strict=False)
        ssl_load_info = {
            "path": args.ssl_checkpoint,
            "missing_keys": missing,
            "unexpected_keys": unexpected,
        }

    model.to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = build_optimizer(
        build_finetune_param_groups(model, args),
        optimizer_name=args.optimizer,
        lr=args.lr,
        weight_decay=args.weight_decay,
        momentum=args.momentum,
    )
    scheduler = build_warmup_cosine_scheduler(
        optimizer,
        epochs=args.epochs,
        warmup_epochs=args.warmup_epochs,
        min_lr_ratio=args.min_lr_ratio,
    )
    use_amp = bool(args.amp and device.type == "cuda")
    scaler = GradScaler("cuda", enabled=use_amp)

    print(f"Dataset root: {args.dataset_root}")
    print(f"Classes: {train_dataset.classes}")
    print(
        f"Train/val/test images: {len(train_dataset)}/"
        f"{len(val_dataset)}/{len(test_dataset)}"
    )
    print(f"Device: {device} | model: {args.model_name} | image_size: {args.image_size}")
    print(f"SSL init: {ssl_load_info['path'] if ssl_load_info else 'none'}")
    print(
        f"Finetune: transform={args.train_transform_mode} lr={args.lr:g} "
        f"backbone_lr={args.lr * args.backbone_lr_mult:g} "
        f"head_lr={args.lr * args.head_lr_mult:g} wd={args.weight_decay:g} "
        f"select={args.selection_metric}"
    )
    print(f"Output: {args.output_dir}")
    if ssl_load_info and ssl_load_info["missing_keys"]:
        print(f"SSL load missing keys: {len(ssl_load_info['missing_keys'])}")

    history: list[dict[str, Any]] = []
    best_selection = initial_best_selection(args.selection_metric)
    best_epoch = -1
    epochs_without_improvement = 0
    start_time = time.time()

    for epoch in range(args.epochs):
        epoch_start = time.time()
        train_metrics = run_train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            scaler,
            use_amp,
            max_steps=1 if args.dry_run else args.max_steps_per_epoch,
        )
        val_metrics = evaluate(
            model,
            val_loader,
            criterion,
            device,
            use_amp,
            num_classes=len(train_dataset.classes),
        )

        selected_value = selection_value(args.selection_metric, val_metrics)
        improved = is_better_selection(args.selection_metric, selected_value, best_selection)
        if improved:
            best_selection = selected_value
            best_epoch = epoch
            epochs_without_improvement = 0
            if not args.dry_run:
                save_model_checkpoint(
                    output_dir / "best_model.pth",
                    model,
                    optimizer,
                    scheduler,
                    epoch,
                    args,
                    train_dataset.classes,
                    val_metrics,
                )
        else:
            epochs_without_improvement += 1

        row = {
            "epoch": epoch + 1,
            "backbone_lr": float(optimizer.param_groups[0]["lr"]),
            "head_lr": float(optimizer.param_groups[1]["lr"]),
            "train_loss": train_metrics["loss"],
            "train_acc": train_metrics["accuracy"],
            "train_macro_f1": train_metrics["macro_f1"],
            "val_loss": val_metrics["loss"],
            "val_acc": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "val_mae": val_metrics["mae"],
            "val_qwk": val_metrics["qwk"],
            "selection_metric": args.selection_metric,
            "selection_value": selected_value,
            "seconds": time.time() - epoch_start,
            "is_best": improved,
        }
        history.append(row)
        print(
            f"Epoch {epoch + 1:03d}/{args.epochs:03d} "
            f"train_acc={train_metrics['accuracy']:.4f} "
            f"val_acc={val_metrics['accuracy']:.4f} "
            f"val_qwk={val_metrics['qwk']:.4f} "
            f"best_{args.selection_metric}={best_selection:.4f}@{best_epoch + 1}"
        )

        if args.dry_run:
            print("Dry run finished after one train step plus validation.")
            return

        scheduler.step()
        write_history_csv(history, output_dir / "history.csv")
        save_model_checkpoint(
            output_dir / "latest_model.pth",
            model,
            optimizer,
            scheduler,
            epoch,
            args,
            train_dataset.classes,
            val_metrics,
        )
        if args.patience > 0 and epochs_without_improvement >= args.patience:
            print(f"Early stopping at epoch {epoch + 1}; no val_acc improvement for {args.patience} epochs.")
            break

    best_checkpoint = torch.load(output_dir / "best_model.pth", map_location=device)
    model.load_state_dict(best_checkpoint["model_state_dict"])
    test_metrics = evaluate(
        model,
        test_loader,
        criterion,
        device,
        use_amp,
        num_classes=len(train_dataset.classes),
    )
    summary = {
        "best_epoch": int(best_checkpoint["epoch"]) + 1,
        "selection_metric": args.selection_metric,
        "best_selection_value": selection_value(args.selection_metric, best_checkpoint["val_metrics"]),
        "best_val_metrics": best_checkpoint["val_metrics"],
        "test_metrics": test_metrics,
        "classes": train_dataset.classes,
        "ssl_load_info": ssl_load_info,
        "total_seconds": time.time() - start_time,
    }
    save_json(summary, output_dir / "summary.json")
    save_json(test_metrics["confusion_matrix"], output_dir / "test_confusion_matrix.json")
    print(
        f"Done. best_epoch={summary['best_epoch']} "
        f"best_{args.selection_metric}={summary['best_selection_value']:.4f} "
        f"best_val_acc={summary['best_val_metrics']['accuracy']:.4f} "
        f"test_acc={test_metrics['accuracy']:.4f} test_qwk={test_metrics['qwk']:.4f}"
    )


if __name__ == "__main__":
    main()
