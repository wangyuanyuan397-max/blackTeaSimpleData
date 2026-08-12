"""Measure parent-internal feature spread against between-stage separation."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from sklearn.metrics import silhouette_score

from audit_common import (
    PatchDataset,
    create_mixnet,
    extract_prelogits,
    load_and_audit_manifest,
    make_loader,
    resolve_device,
    seed_everything,
    write_csv,
    write_json,
)


TEMP_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = TEMP_ROOT.parents[1]
DEFAULT_DATASET_ROOT = PROJECT_ROOT / "datasets_01234_BaSic"
DEFAULT_MANIFEST = DEFAULT_DATASET_ROOT / "grid30_crop_manifest.csv"
DEFAULT_OUTPUT_DIR = TEMP_ROOT / "results" / "feature_audit"
TIME_CODES = ("00", "10", "20", "30", "40")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MixNet-S parent-vs-class feature geometry audit.")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--image-size", type=int, default=408)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--random-backbone", action="store_true", help="Use random weights; only useful as a smoke test/control.")
    return parser.parse_args()


def load_checkpoint(model: torch.nn.Module, path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        raise RuntimeError(
            "Checkpoint is not a strict timm MixNet-S five-class state. "
            "Use best_model.pth produced by run_audit.py M01 or a matching checkpoint."
        ) from exc
    return checkpoint if isinstance(checkpoint, dict) else {}


def extract_features(
    model: torch.nn.Module,
    dataset: PatchDataset,
    *,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    loader = make_loader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        seed=seed,
    )
    features: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    model.eval()
    with torch.inference_mode():
        for images, targets, _ in loader:
            embedding = extract_prelogits(model, images.to(device, non_blocking=True))
            embedding = torch.nn.functional.normalize(embedding, dim=1)
            features.append(embedding.cpu().numpy())
            labels.append(targets.numpy())
    return np.concatenate(features), np.concatenate(labels)


def cosine_distance(left: np.ndarray, right: np.ndarray) -> float:
    left = left / max(float(np.linalg.norm(left)), 1e-12)
    right = right / max(float(np.linalg.norm(right)), 1e-12)
    return float(1.0 - np.dot(left, right))


def bootstrap_ci(values: Sequence[float], *, iterations: int, seed: int) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    if len(array) < 2 or iterations <= 0:
        value = float(array.mean()) if len(array) else float("nan")
        return value, value
    generator = np.random.default_rng(seed)
    means = np.empty(iterations, dtype=np.float64)
    for index in range(iterations):
        means[index] = generator.choice(array, size=len(array), replace=True).mean()
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def main() -> None:
    args = parse_args()
    if args.checkpoint is None and not args.random_backbone:
        raise ValueError("Provide --checkpoint from M01, or explicitly use --random-backbone for a control.")
    seed_everything(args.seed)
    device = resolve_device(args.device)
    samples, manifest_audit = load_and_audit_manifest(args.manifest, args.dataset_root)
    dataset = PatchDataset(
        samples,
        split=args.split,
        time_codes=TIME_CODES,
        image_size=args.image_size,
        training=False,
    )
    model = create_mixnet(len(TIME_CODES), pretrained=False).to(device)
    checkpoint_meta: dict[str, Any] = {}
    if args.checkpoint is not None:
        checkpoint_meta = load_checkpoint(model, args.checkpoint)
    features, labels = extract_features(
        model,
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
        seed=args.seed,
    )

    grouped_indices: dict[str, list[int]] = defaultdict(list)
    for index, sample in enumerate(dataset.samples):
        grouped_indices[sample.source_image_id].append(index)

    parent_rows: list[dict[str, Any]] = []
    parent_centroids: dict[str, np.ndarray] = {}
    class_parent_centroids: dict[int, list[np.ndarray]] = defaultdict(list)
    for parent_id in sorted(grouped_indices):
        indices = grouped_indices[parent_id]
        local = features[indices]
        centroid = local.mean(axis=0)
        centroid /= max(float(np.linalg.norm(centroid)), 1e-12)
        distances = np.asarray([cosine_distance(vector, centroid) for vector in local])
        label = int(labels[indices[0]])
        parent_centroids[parent_id] = centroid
        class_parent_centroids[label].append(centroid)
        parent_rows.append(
            {
                "source_image_id": parent_id,
                "time_code": TIME_CODES[label],
                "patch_count": len(indices),
                "within_cosine_mean": float(distances.mean()),
                "within_cosine_rms": float(np.sqrt(np.mean(distances**2))),
                "within_cosine_p90": float(np.quantile(distances, 0.90)),
            }
        )

    class_centroids: dict[int, np.ndarray] = {}
    for label, centroids in class_parent_centroids.items():
        centroid = np.vstack(centroids).mean(axis=0)
        class_centroids[label] = centroid / max(float(np.linalg.norm(centroid)), 1e-12)

    pair_rows: list[dict[str, Any]] = []
    for left in range(len(TIME_CODES)):
        for right in range(left + 1, len(TIME_CODES)):
            distance = cosine_distance(class_centroids[left], class_centroids[right])
            relevant_within = [
                row["within_cosine_mean"]
                for row in parent_rows
                if row["time_code"] in {TIME_CODES[left], TIME_CODES[right]}
            ]
            within_mean = float(np.mean(relevant_within))
            ratios = [value / max(distance, 1e-12) for value in relevant_within]
            ci_low, ci_high = bootstrap_ci(ratios, iterations=args.bootstrap, seed=args.seed + left * 10 + right)
            pair_rows.append(
                {
                    "left_time": TIME_CODES[left],
                    "right_time": TIME_CODES[right],
                    "is_adjacent": int(right == left + 1),
                    "class_centroid_cosine_distance": distance,
                    "mean_parent_within_cosine": within_mean,
                    "within_to_between_ratio": within_mean / max(distance, 1e-12),
                    "ratio_parent_bootstrap_ci_low": ci_low,
                    "ratio_parent_bootstrap_ci_high": ci_high,
                    "interpretation": (
                        "within_exceeds_between"
                        if within_mean > distance
                        else "within_at_least_half_between"
                        if within_mean >= 0.5 * distance
                        else "between_dominates"
                    ),
                }
            )

    class_rows: list[dict[str, Any]] = []
    for label, time_code in enumerate(TIME_CODES):
        rows = [row for row in parent_rows if row["time_code"] == time_code]
        values = [row["within_cosine_mean"] for row in rows]
        ci_low, ci_high = bootstrap_ci(values, iterations=args.bootstrap, seed=args.seed + label)
        class_rows.append(
            {
                "time_code": time_code,
                "parent_count": len(rows),
                "within_cosine_mean": float(np.mean(values)),
                "within_cosine_median": float(np.median(values)),
                "within_cosine_ci_low": ci_low,
                "within_cosine_ci_high": ci_high,
            }
        )

    silhouette = float(silhouette_score(features, labels, metric="cosine"))
    summary = {
        "split": args.split,
        "checkpoint": str(args.checkpoint) if args.checkpoint else None,
        "random_backbone": bool(args.random_backbone),
        "sample_count": len(dataset),
        "parent_count": len(parent_rows),
        "feature_dimension": int(features.shape[1]),
        "cosine_silhouette_patch_level": silhouette,
        "manifest_audit": manifest_audit,
        "checkpoint_best_epoch": checkpoint_meta.get("best_epoch"),
        "primary_quantity": "mean cosine distance of each patch to its parent centroid / cosine distance between class centroids",
        "important_limit": "The split has few independent parents; interpret bootstrap intervals and repeat across seeds.",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "parent_within_spread.csv", parent_rows)
    write_csv(args.output_dir / "class_within_spread.csv", class_rows)
    write_csv(args.output_dir / "within_vs_between.csv", pair_rows)
    write_json(args.output_dir / "feature_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Feature audit written to: {args.output_dir}")


if __name__ == "__main__":
    main()
