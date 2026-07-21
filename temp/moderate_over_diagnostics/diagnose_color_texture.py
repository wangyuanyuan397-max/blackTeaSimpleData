"""Diagnose color-only and texture-only separability with simple SVM features."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from common import compute_binary_metrics, ensure_dir, patch_metadata_path, save_confusion_matrix_png, save_json, source_splits_path


def load_patch_frame(run_dir: Path) -> pd.DataFrame:
    splits = pd.read_csv(source_splits_path(run_dir), dtype={"time_code": str})[["source_image_id", "split"]]
    frame = pd.read_csv(patch_metadata_path(run_dir), dtype={"time_code": str})
    frame = frame.merge(splits, on="source_image_id", how="inner")
    frame["image_path"] = frame["patch_path"]
    frame["label"] = frame["label"].astype(int)
    return frame.reset_index(drop=True)


def load_rgb(path: str, image_size: int) -> np.ndarray:
    image = Image.open(path)
    image = ImageOps.exif_transpose(image).convert("RGB")
    image = image.resize((image_size, image_size), Image.Resampling.BICUBIC)
    return np.asarray(image, dtype=np.uint8)


def normalize_hist(hist: np.ndarray) -> np.ndarray:
    hist = hist.astype(np.float32)
    return hist / max(1.0, float(hist.sum()))


def channel_hist(array: np.ndarray, bins: int, value_range=(0, 256)) -> np.ndarray:
    features = []
    for channel in range(array.shape[2]):
        hist, _ = np.histogram(array[..., channel], bins=bins, range=value_range)
        features.append(normalize_hist(hist))
    return np.concatenate(features)


def lab_like(rgb: np.ndarray) -> np.ndarray:
    arr = rgb.astype(np.float32) / 255.0
    mask = arr > 0.04045
    linear = np.where(mask, ((arr + 0.055) / 1.055) ** 2.4, arr / 12.92)
    matrix = np.array([[0.4124564, 0.3575761, 0.1804375], [0.2126729, 0.7151522, 0.0721750], [0.0193339, 0.1191920, 0.9503041]], dtype=np.float32)
    xyz = linear @ matrix.T
    xyz = xyz / np.array([0.95047, 1.0, 1.08883], dtype=np.float32)
    delta = 6 / 29
    f = np.where(xyz > delta ** 3, np.cbrt(xyz), xyz / (3 * delta ** 2) + 4 / 29)
    lab = np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], axis=2)
    lab[..., 0] = np.clip(lab[..., 0] / 100.0 * 255.0, 0, 255)
    lab[..., 1] = np.clip(lab[..., 1] + 128.0, 0, 255)
    lab[..., 2] = np.clip(lab[..., 2] + 128.0, 0, 255)
    return lab.astype(np.uint8)


def lbp_hist(rgb: np.ndarray) -> np.ndarray:
    gray = np.asarray(Image.fromarray(rgb).convert("L"), dtype=np.uint8)
    center = gray[1:-1, 1:-1]
    codes = np.zeros_like(center, dtype=np.uint8)
    offsets = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]
    for bit, (dy, dx) in enumerate(offsets):
        neighbor = gray[1 + dy: gray.shape[0] - 1 + dy, 1 + dx: gray.shape[1] - 1 + dx]
        codes |= ((neighbor >= center).astype(np.uint8) << bit)
    hist, _ = np.histogram(codes, bins=256, range=(0, 256))
    return normalize_hist(hist)


def glcm_stats(rgb: np.ndarray, levels: int = 16) -> np.ndarray:
    gray = np.asarray(Image.fromarray(rgb).convert("L"), dtype=np.uint8)
    quantized = np.clip((gray.astype(np.int32) * levels) // 256, 0, levels - 1)
    ij = np.arange(levels)
    i_grid, j_grid = np.meshgrid(ij, ij, indexing="ij")
    features = []
    for dy, dx in [(0, 1), (1, 0), (1, 1), (-1, 1)]:
        src = quantized[max(0, -dy): quantized.shape[0] - max(0, dy), max(0, -dx): quantized.shape[1] - max(0, dx)]
        dst = quantized[max(0, dy): quantized.shape[0] - max(0, -dy), max(0, dx): quantized.shape[1] - max(0, -dx)]
        matrix = np.zeros((levels, levels), dtype=np.float32)
        np.add.at(matrix, (src.ravel(), dst.ravel()), 1)
        matrix = matrix / max(1.0, float(matrix.sum()))
        contrast = ((i_grid - j_grid) ** 2 * matrix).sum()
        homogeneity = (matrix / (1.0 + np.abs(i_grid - j_grid))).sum()
        energy = np.sqrt((matrix ** 2).sum())
        entropy = -(matrix[matrix > 0] * np.log2(matrix[matrix > 0])).sum()
        features.extend([contrast, homogeneity, energy, entropy])
    return np.asarray(features, dtype=np.float32)


def feature_one(path: str, feature_name: str, image_size: int, bins: int) -> np.ndarray:
    rgb = load_rgb(path, image_size)
    if feature_name == "rgb_hist":
        return channel_hist(rgb, bins)
    if feature_name == "hsv_hist":
        hsv = np.asarray(Image.fromarray(rgb).convert("HSV"), dtype=np.uint8)
        return channel_hist(hsv, bins)
    if feature_name == "lab_hist":
        return channel_hist(lab_like(rgb), bins)
    if feature_name == "lbp":
        return lbp_hist(rgb)
    if feature_name == "glcm":
        return glcm_stats(rgb)
    if feature_name == "color_texture_concat":
        return np.concatenate([channel_hist(np.asarray(Image.fromarray(rgb).convert("HSV"), dtype=np.uint8), bins), lbp_hist(rgb), glcm_stats(rgb)])
    raise ValueError(f"Unknown feature: {feature_name}")


def extract_matrix(frame: pd.DataFrame, feature_name: str, image_size: int, bins: int) -> np.ndarray:
    return np.vstack([feature_one(path, feature_name, image_size, bins) for path in frame["image_path"]]).astype(np.float32)


def plot_feature_summary(summary: pd.DataFrame, output_path: Path) -> None:
    ordered = summary.sort_values("macro_f1", ascending=True)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.barh(ordered["feature"], ordered["macro_f1"], color="#4C78A8")
    ax.set_xlim(0, 1)
    ax.set_xlabel("Macro-F1")
    ax.set_title("Color / texture feature separability")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def diagnose(args: argparse.Namespace) -> Path:
    run_dir = Path(args.run_dir)
    out_dir = ensure_dir(run_dir / "color_texture")
    frame = load_patch_frame(run_dir)
    train_val = frame[frame["split"].isin(["train", "val"])].reset_index(drop=True)
    test = frame[frame["split"] == "test"].reset_index(drop=True)
    y_train = train_val["label"].to_numpy(dtype=np.int64)
    y_test = test["label"].to_numpy(dtype=np.int64)
    feature_names = ["rgb_hist", "hsv_hist", "lab_hist", "lbp", "glcm", "color_texture_concat"]
    all_metrics = {}
    rows = []
    for feature_name in feature_names:
        print(f"evaluating {feature_name}")
        x_train = extract_matrix(train_val, feature_name, args.image_size, args.bins)
        x_test = extract_matrix(test, feature_name, args.image_size, args.bins)
        classifier = make_pipeline(StandardScaler(), SVC(kernel="rbf", C=args.svm_c, gamma="scale", class_weight="balanced"))
        classifier.fit(x_train, y_train)
        pred = classifier.predict(x_test).astype(int)
        metrics = compute_binary_metrics(y_test, pred)
        all_metrics[feature_name] = metrics
        rows.append({"feature": feature_name, "accuracy": metrics["accuracy"], "macro_f1": metrics["macro_f1"]})
        pred_frame = test.copy()
        pred_frame["pred_label"] = pred
        pred_frame.to_csv(out_dir / f"{feature_name}_test_predictions.csv", index=False, encoding="utf-8-sig")
        save_confusion_matrix_png(metrics["confusion_matrix"], out_dir / f"{feature_name}_confusion_matrix.png", feature_name)
    summary = pd.DataFrame(rows).sort_values("macro_f1", ascending=False)
    summary.to_csv(out_dir / "feature_summary.csv", index=False, encoding="utf-8-sig")
    plot_feature_summary(summary, out_dir / "feature_summary.png")
    save_json(out_dir / "feature_metrics.json", all_metrics)
    print(f"color/texture diagnostics saved: {out_dir}")
    return out_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run color/texture feature diagnostics with SVM.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--bins", type=int, default=32)
    parser.add_argument("--svm-c", type=float, default=3.0)
    return parser.parse_args()


if __name__ == "__main__":
    diagnose(parse_args())
