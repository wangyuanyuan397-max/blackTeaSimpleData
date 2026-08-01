"""Self-supervised SimSiam pretraining for MixNet-S on BaSiC tea patches."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from common import (
    PROJECT_ROOT,
    SimSiamModel,
    TwoViewImageDataset,
    build_optimizer,
    build_ssl_transform,
    build_warmup_cosine_scheduler,
    list_image_files,
    make_worker_init_fn,
    resolve_project_path,
    save_json,
    seed_everything,
    simsiam_loss,
    write_history_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run SimSiam self-supervised pretraining for MixNet-S."
    )
    parser.add_argument("--dataset-root", default="datasets_01234_BaSic")
    parser.add_argument(
        "--ssl-splits",
        nargs="+",
        default=["train"],
        help="Dataset split folders to use without labels. Default avoids val/test leakage.",
    )
    parser.add_argument("--model-name", default="mixnet_s")
    parser.add_argument("--image-size", type=int, default=408)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--optimizer", choices=["adamw", "adam", "sgd"], default="adamw")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--min-lr-ratio", type=float, default=0.01)
    parser.add_argument("--crop-min", type=float, default=0.55)
    parser.add_argument("--jitter-strength", type=float, default=0.35)
    parser.add_argument("--projection-dim", type=int, default=2048)
    parser.add_argument("--projection-hidden-dim", type=int, default=2048)
    parser.add_argument("--prediction-hidden-dim", type=int, default=512)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--save-every", type=int, default=25)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--max-steps-per-epoch", type=int, default=0)
    parser.add_argument("--imagenet-pretrained", action="store_true")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--output-dir",
        default="temp/mixnet_simsiam_ssl/runs/simsiam_mixnet_s_basic408",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run one forward/backward batch and exit without writing checkpoints.",
    )
    return parser.parse_args()


def build_ssl_roots(dataset_root: Path, splits: list[str]) -> list[Path]:
    roots: list[Path] = []
    for split in splits:
        if split in {".", "root"}:
            split_root = dataset_root
        else:
            split_root = dataset_root / split
        if not split_root.is_dir():
            raise FileNotFoundError(f"SSL split folder does not exist: {split_root}")
        roots.append(split_root)
    return roots


def save_checkpoint(
    output_dir: Path,
    filename: str,
    model: SimSiamModel,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    epoch: int,
    args: argparse.Namespace,
    history: list[dict],
) -> Path:
    checkpoint_path = output_dir / filename
    torch.save(
        {
            "epoch": epoch,
            "model_name": args.model_name,
            "image_size": args.image_size,
            "backbone_state_dict": model.backbone.state_dict(),
            "simsiam_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "args": vars(args),
            "history": history,
        },
        checkpoint_path,
    )
    return checkpoint_path


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    dataset_root = resolve_project_path(args.dataset_root)
    output_dir = resolve_project_path(args.output_dir)
    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
        save_json(vars(args), output_dir / "args.json")

    ssl_roots = build_ssl_roots(dataset_root, args.ssl_splits)
    image_paths = list_image_files(ssl_roots)
    if args.max_samples > 0:
        image_paths = image_paths[: args.max_samples]

    if len(image_paths) < 2:
        raise ValueError("SimSiam needs at least 2 images.")

    transform = build_ssl_transform(
        image_size=args.image_size,
        crop_min=args.crop_min,
        jitter_strength=args.jitter_strength,
    )
    dataset = TwoViewImageDataset(image_paths, transform=transform)
    drop_last = len(dataset) >= args.batch_size
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=drop_last,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=bool(args.num_workers > 0),
        worker_init_fn=make_worker_init_fn(args.seed),
        generator=generator,
    )
    if len(loader) == 0:
        raise ValueError("No batches were produced. Lower --batch-size or add images.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimSiamModel(
        model_name=args.model_name,
        pretrained=args.imagenet_pretrained,
        image_size=args.image_size,
        projection_dim=args.projection_dim,
        projection_hidden_dim=args.projection_hidden_dim,
        prediction_hidden_dim=args.prediction_hidden_dim,
    ).to(device)
    optimizer = build_optimizer(
        model.parameters(),
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

    print(f"Project root: {PROJECT_ROOT}")
    print(f"SSL image roots: {[str(root) for root in ssl_roots]}")
    print(f"Images: {len(dataset)} | batches/epoch: {len(loader)}")
    print(f"Device: {device} | model: {args.model_name} | image_size: {args.image_size}")
    print(f"Output: {output_dir}")

    history: list[dict] = []
    start_time = time.time()
    for epoch in range(args.epochs):
        model.train()
        epoch_loss_sum = 0.0
        sample_count = 0
        step_count = 0
        epoch_start = time.time()

        for step, (x1, x2, _paths) in enumerate(loader, start=1):
            x1 = x1.to(device, non_blocking=True)
            x2 = x2.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type=device.type, enabled=use_amp):
                p1, z1, p2, z2 = model(x1, x2)
                loss = simsiam_loss(p1, z1, p2, z2)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            batch_size = int(x1.size(0))
            epoch_loss_sum += float(loss.detach().item()) * batch_size
            sample_count += batch_size
            step_count += 1

            if args.dry_run:
                break
            if args.max_steps_per_epoch > 0 and step >= args.max_steps_per_epoch:
                break

        epoch_loss = epoch_loss_sum / max(1, sample_count)
        current_lr = float(optimizer.param_groups[0]["lr"])
        row = {
            "epoch": epoch + 1,
            "loss": epoch_loss,
            "lr": current_lr,
            "samples": sample_count,
            "steps": step_count,
            "seconds": time.time() - epoch_start,
        }
        history.append(row)
        print(
            f"Epoch {epoch + 1:03d}/{args.epochs:03d} "
            f"loss={epoch_loss:.5f} lr={current_lr:.6g} steps={step_count}"
        )

        if args.dry_run:
            print("Dry run finished after one optimization step.")
            return

        scheduler.step()
        write_history_csv(history, output_dir / "history.csv")
        save_checkpoint(output_dir, "latest_checkpoint.pth", model, optimizer, scheduler, epoch, args, history)
        if args.save_every > 0 and (epoch + 1) % args.save_every == 0:
            save_checkpoint(
                output_dir,
                f"checkpoint_epoch_{epoch + 1:03d}.pth",
                model,
                optimizer,
                scheduler,
                epoch,
                args,
                history,
            )

    final_backbone_path = output_dir / f"{args.model_name}_simsiam_backbone.pth"
    torch.save(
        {
            "model_name": args.model_name,
            "image_size": args.image_size,
            "backbone_state_dict": model.backbone.state_dict(),
            "args": vars(args),
            "history": history,
            "total_seconds": time.time() - start_time,
        },
        final_backbone_path,
    )
    save_checkpoint(output_dir, "final_checkpoint.pth", model, optimizer, scheduler, args.epochs - 1, args, history)
    print(f"Saved backbone weights: {final_backbone_path}")


if __name__ == "__main__":
    main()
