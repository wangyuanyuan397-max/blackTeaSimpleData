"""分析原图 HSV/Lab 分布，检验类别 20 的 ``4-*`` 是否存在真实颜色偏移。

本文件专为 PyCharm 右键运行设计，不读取命令行参数。所有可修改参数都集中在
下方 ``HSVLabAnalysisConfig``。统计推断以“原图”为独立样本，绝不把像素数量
当作样本量；像素仅用于估计每张图自身的颜色分布。
"""

from __future__ import annotations

import csv
import itertools
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import cv2
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist, jensenshannon


EXPERIMENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EXPERIMENT_DIR.parents[1]


# =============================================================================
# PyCharm 右键运行配置区：不需要、也不接受命令行参数
# =============================================================================
@dataclass(frozen=True)
class HSVLabAnalysisConfig:
    """HSV/Lab 原图分析配置。"""

    manifest_path: Path = (
        PROJECT_ROOT / "datasets_01234_original_split" / "source_split_manifest.csv"
    )
    output_dir: Path = EXPERIMENT_DIR / "results" / "hsv_lab_source_shift"

    # 主分析类别和待检验的来源组。
    target_class: str = "20"
    target_source_group: str = "4"

    # 每隔若干个原始像素取一个点。固定网格只降低计算量，不改变图像颜色。
    pixel_stride: int = 4

    # 图像级 bootstrap 次数；精确置换检验会穷举所有 C(19, 4)=3876 种分组。
    bootstrap_iterations: int = 10_000
    random_seed: int = 2026

    # 期刊双栏宽度约 183 mm；SVG/PDF 保留可编辑文字。
    figure_width_mm: float = 183.0
    distribution_height_mm: float = 118.0
    summary_height_mm: float = 142.0
    preview_dpi: int = 300


CONFIG = HSVLabAnalysisConfig()


GROUP_ORDER = ("train_20", "target_20_4", "other_holdout_20")
GROUP_LABELS = {
    "train_20": "Train class 20",
    "target_20_4": "20/4-*",
    "other_holdout_20": "Other held-out 20",
}
GROUP_COLORS = {
    "train_20": "#64748B",
    "target_20_4": "#D95F59",
    "other_holdout_20": "#3A8D8A",
}


# 六个主通道的直方图定义。Hue 采用饱和度加权，降低近灰像素的不稳定色相。
HISTOGRAM_SPECS: dict[str, tuple[str, int, tuple[float, float]]] = {
    "hue_deg": ("Hue (degrees)", 36, (0.0, 360.0)),
    "saturation": ("Saturation", 32, (0.0, 1.0)),
    "value": ("Value", 32, (0.0, 1.0)),
    "lab_l": ("L*", 32, (0.0, 100.0)),
    "lab_a": ("a*", 40, (-40.0, 40.0)),
    "lab_b": ("b*", 40, (-40.0, 40.0)),
}


# 主检验特征。Hue 是圆周变量，单独使用圆周均值差检验。
TEST_FEATURES = (
    ("hue_circular_mean_deg", "Hue", True),
    ("saturation_mean", "Saturation", False),
    ("value_mean", "Value", False),
    ("lab_l_mean", "L*", False),
    ("lab_a_mean", "a*", False),
    ("lab_b_mean", "b*", False),
    ("lab_chroma_mean", "Chroma C*", False),
)


def read_manifest(path: Path) -> list[dict[str, str]]:
    """读取原图拆分清单。"""

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """保存 UTF-8-BOM CSV，便于 Excel 直接打开中文。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8-sig")
        return
    fieldnames: list[str] = []
    for row in rows:
        fieldnames.extend(key for key in row if key not in fieldnames)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def source_group(source_stem: str) -> str:
    """返回文件名连字符前的来源组，例如 ``4-2`` 返回 ``4``。"""

    return source_stem.split("-", maxsplit=1)[0]


def source_index(source_stem: str) -> str:
    """返回文件名连字符后的组内编号，例如 ``4-2`` 返回 ``2``。"""

    parts = source_stem.split("-", maxsplit=1)
    return parts[1] if len(parts) == 2 else ""


def classify_group(row: dict[str, str], config: HSVLabAnalysisConfig) -> str | None:
    """把类别 20 原图分成训练、目标来源组和其他留出图。"""

    if row["time_code"] != config.target_class:
        return None
    group = source_group(row["source_stem"])
    if group == config.target_source_group:
        return "target_20_4"
    if row["split"] == "train":
        return "train_20"
    return "other_holdout_20"


def weighted_circular_mean_deg(hue_deg: np.ndarray, weights: np.ndarray) -> float:
    """计算饱和度加权的圆周色相均值。"""

    angles = np.deg2rad(hue_deg)
    safe_weights = np.maximum(weights.astype(np.float64), 1e-8)
    x = float(np.average(np.cos(angles), weights=safe_weights))
    y = float(np.average(np.sin(angles), weights=safe_weights))
    return float(np.rad2deg(math.atan2(y, x)) % 360.0)


def circular_difference_deg(angle_a: float, angle_b: float) -> float:
    """返回 angle_a-angle_b 的最短有符号圆周差，范围 [-180, 180)。"""

    return float((angle_a - angle_b + 180.0) % 360.0 - 180.0)


def normalized_histogram(
    values: np.ndarray,
    bins: int,
    value_range: tuple[float, float],
    weights: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """计算总和为 1 的概率直方图。"""

    counts, edges = np.histogram(values, bins=bins, range=value_range, weights=weights)
    counts = counts.astype(np.float64)
    total = float(counts.sum())
    if total <= 0:
        raise ValueError("颜色直方图权重总和为 0，无法归一化。")
    return counts / total, edges.astype(np.float64)


def load_color_arrays(path: Path, stride: int) -> dict[str, np.ndarray]:
    """从原图固定网格取样，并转换成标准范围的 HSV 与 CIELAB。"""

    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"无法读取原图：{path}")
    sampled = bgr[::stride, ::stride]

    # OpenCV 的 HSV 色相范围是 0~179；转换为角度 0~358。
    hsv = cv2.cvtColor(sampled, cv2.COLOR_BGR2HSV).astype(np.float32)
    hue_deg = hsv[..., 0].reshape(-1) * 2.0
    saturation = hsv[..., 1].reshape(-1) / 255.0
    value = hsv[..., 2].reshape(-1) / 255.0

    # OpenCV 8 位 Lab：L 映射到 0~255，a/b 加了 128 偏移。
    lab = cv2.cvtColor(sampled, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab_l = lab[..., 0].reshape(-1) * (100.0 / 255.0)
    lab_a = lab[..., 1].reshape(-1) - 128.0
    lab_b = lab[..., 2].reshape(-1) - 128.0
    chroma = np.sqrt(np.square(lab_a) + np.square(lab_b))

    return {
        "hue_deg": hue_deg,
        "saturation": saturation,
        "value": value,
        "lab_l": lab_l,
        "lab_a": lab_a,
        "lab_b": lab_b,
        "lab_chroma": chroma,
    }


def summarize_image(
    row: dict[str, str],
    group: str,
    arrays: dict[str, np.ndarray],
) -> dict[str, Any]:
    """把像素颜色压缩为一行图像级特征。"""

    hue = arrays["hue_deg"]
    saturation = arrays["saturation"]
    result: dict[str, Any] = {
        "group": group,
        "group_label": GROUP_LABELS[group],
        "split": row["split"],
        "class_label": row["time_code"],
        "source_stem": row["source_stem"],
        "source_group": source_group(row["source_stem"]),
        "source_index": source_index(row["source_stem"]),
        "source_path": row["source_path"],
        "sampled_pixel_count": int(hue.size),
        "hue_circular_mean_deg": weighted_circular_mean_deg(hue, saturation),
    }
    for key in ("saturation", "value", "lab_l", "lab_a", "lab_b", "lab_chroma"):
        values = arrays[key]
        result[f"{key}_mean"] = float(np.mean(values))
        result[f"{key}_median"] = float(np.median(values))
        result[f"{key}_std"] = float(np.std(values, ddof=0))
    return result


def build_histogram_rows(
    image_row: dict[str, Any],
    arrays: dict[str, np.ndarray],
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    """生成长表直方图及供距离计算使用的向量。"""

    rows: list[dict[str, Any]] = []
    vectors: dict[str, np.ndarray] = {}
    for channel, (label, bins, value_range) in HISTOGRAM_SPECS.items():
        weights = arrays["saturation"] if channel == "hue_deg" else None
        density, edges = normalized_histogram(
            arrays[channel], bins=bins, value_range=value_range, weights=weights
        )
        vectors[channel] = density
        for index, probability in enumerate(density):
            rows.append(
                {
                    "group": image_row["group"],
                    "group_label": image_row["group_label"],
                    "split": image_row["split"],
                    "source_stem": image_row["source_stem"],
                    "channel": channel,
                    "channel_label": label,
                    "bin_index": index,
                    "bin_left": float(edges[index]),
                    "bin_right": float(edges[index + 1]),
                    "bin_center": float((edges[index] + edges[index + 1]) / 2.0),
                    "probability": float(probability),
                }
            )
    return rows, vectors


def circular_group_mean(values: np.ndarray) -> float:
    """计算一组图像级色相均值的圆周均值。"""

    angles = np.deg2rad(values)
    return float(np.rad2deg(math.atan2(np.mean(np.sin(angles)), np.mean(np.cos(angles)))) % 360.0)


def exact_permutation_test(
    target_values: np.ndarray,
    train_values: np.ndarray,
    statistic: Callable[[np.ndarray, np.ndarray], float],
) -> tuple[float, int]:
    """穷举所有同样本量分组，返回双侧/非负统计量的精确 p 值。"""

    pooled = np.concatenate([target_values, train_values])
    target_n = len(target_values)
    observed = abs(float(statistic(target_values, train_values)))
    exceed = 0
    permutations = 0
    all_indices = np.arange(len(pooled))
    for chosen in itertools.combinations(range(len(pooled)), target_n):
        mask = np.zeros(len(pooled), dtype=bool)
        mask[list(chosen)] = True
        candidate = abs(float(statistic(pooled[mask], pooled[~mask])))
        exceed += int(candidate >= observed - 1e-12)
        permutations += 1
    return exceed / permutations, permutations


def hedges_g(target: np.ndarray, train: np.ndarray) -> float:
    """计算小样本校正后的标准化均值差 Hedges' g。"""

    n_target, n_train = len(target), len(train)
    pooled_variance = (
        (n_target - 1) * np.var(target, ddof=1)
        + (n_train - 1) * np.var(train, ddof=1)
    ) / (n_target + n_train - 2)
    if pooled_variance <= 0:
        return 0.0
    cohen_d = (float(np.mean(target)) - float(np.mean(train))) / math.sqrt(pooled_variance)
    correction = 1.0 - 3.0 / (4.0 * (n_target + n_train) - 9.0)
    return float(correction * cohen_d)


def cliffs_delta(target: np.ndarray, train: np.ndarray) -> float:
    """计算 Cliff's delta，正值表示目标组整体更高。"""

    differences = target[:, None] - train[None, :]
    return float((np.sum(differences > 0) - np.sum(differences < 0)) / differences.size)


def bootstrap_difference(
    target: np.ndarray,
    train: np.ndarray,
    statistic: Callable[[np.ndarray, np.ndarray], float],
    iterations: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    """按图像重采样，估计组间差异的百分位数 95% CI。"""

    estimates = np.empty(iterations, dtype=np.float64)
    for index in range(iterations):
        target_boot = rng.choice(target, size=len(target), replace=True)
        train_boot = rng.choice(train, size=len(train), replace=True)
        estimates[index] = statistic(target_boot, train_boot)
    low, high = np.quantile(estimates, [0.025, 0.975])
    return float(low), float(high)


def benjamini_hochberg(p_values: list[float]) -> list[float]:
    """Benjamini-Hochberg 多重比较校正。"""

    values = np.asarray(p_values, dtype=np.float64)
    order = np.argsort(values)
    ranked = values[order]
    adjusted = ranked * len(values) / np.arange(1, len(values) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)
    restored = np.empty_like(adjusted)
    restored[order] = adjusted
    return restored.tolist()


def compute_feature_tests(
    feature_rows: list[dict[str, Any]],
    config: HSVLabAnalysisConfig,
) -> list[dict[str, Any]]:
    """逐特征比较目标组与训练组，并进行精确置换检验。"""

    rng = np.random.default_rng(config.random_seed)
    tests: list[dict[str, Any]] = []
    for feature, label, is_circular in TEST_FEATURES:
        target = np.asarray(
            [row[feature] for row in feature_rows if row["group"] == "target_20_4"],
            dtype=np.float64,
        )
        train = np.asarray(
            [row[feature] for row in feature_rows if row["group"] == "train_20"],
            dtype=np.float64,
        )
        if is_circular:
            difference_function = lambda a, b: circular_difference_deg(
                circular_group_mean(a), circular_group_mean(b)
            )
            effect = float("nan")
            delta = float("nan")
            target_center = circular_group_mean(target)
            train_center = circular_group_mean(train)
        else:
            difference_function = lambda a, b: float(np.mean(a) - np.mean(b))
            effect = hedges_g(target, train)
            delta = cliffs_delta(target, train)
            target_center = float(np.mean(target))
            train_center = float(np.mean(train))

        observed_difference = float(difference_function(target, train))
        p_value, permutation_count = exact_permutation_test(
            target, train, difference_function
        )
        ci_low, ci_high = bootstrap_difference(
            target,
            train,
            difference_function,
            config.bootstrap_iterations,
            rng,
        )
        tests.append(
            {
                "feature": feature,
                "feature_label": label,
                "target_n": len(target),
                "train_n": len(train),
                "target_center": target_center,
                "train_center": train_center,
                "target_minus_train": observed_difference,
                "bootstrap_ci95_low": ci_low,
                "bootstrap_ci95_high": ci_high,
                "hedges_g": effect,
                "cliffs_delta": delta,
                "exact_permutation_p": p_value,
                "permutation_count": permutation_count,
                "circular_feature": is_circular,
            }
        )
    q_values = benjamini_hochberg([row["exact_permutation_p"] for row in tests])
    for row, q_value in zip(tests, q_values):
        row["bh_q"] = q_value
    return tests


def energy_distance_statistic(x: np.ndarray, y: np.ndarray) -> float:
    """计算两个多维图像组之间的样本能量距离。"""

    between = np.mean(cdist(x, y, metric="euclidean"))
    within_x = np.mean(cdist(x, x, metric="euclidean"))
    within_y = np.mean(cdist(y, y, metric="euclidean"))
    return float(2.0 * between - within_x - within_y)


def compute_multivariate_test(feature_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """对 HSV/Lab 联合特征进行精确多元置换检验。"""

    selected = [row for row in feature_rows if row["group"] in {"target_20_4", "train_20"}]
    features = []
    labels = []
    for row in selected:
        hue_rad = math.radians(float(row["hue_circular_mean_deg"]))
        features.append(
            [
                math.cos(hue_rad),
                math.sin(hue_rad),
                row["saturation_mean"],
                row["value_mean"],
                row["lab_l_mean"],
                row["lab_a_mean"],
                row["lab_b_mean"],
                row["lab_chroma_mean"],
            ]
        )
        labels.append(row["group"])
    matrix = np.asarray(features, dtype=np.float64)

    # 用合并样本的标准差固定尺度；因为尺度与标签置换无关，精确检验仍然有效。
    scale = np.std(matrix, axis=0, ddof=1)
    scale[scale == 0] = 1.0
    standardized = (matrix - np.mean(matrix, axis=0)) / scale
    target_mask = np.asarray([label == "target_20_4" for label in labels])
    target_n = int(np.sum(target_mask))
    observed = energy_distance_statistic(standardized[target_mask], standardized[~target_mask])

    exceed = 0
    permutation_count = 0
    for chosen in itertools.combinations(range(len(labels)), target_n):
        mask = np.zeros(len(labels), dtype=bool)
        mask[list(chosen)] = True
        candidate = energy_distance_statistic(standardized[mask], standardized[~mask])
        exceed += int(candidate >= observed - 1e-12)
        permutation_count += 1
    return {
        "test": "standardized_multivariate_energy_distance",
        "features": "hue_cos+hue_sin+saturation+value+L+a+b+chroma",
        "target_n": target_n,
        "train_n": len(labels) - target_n,
        "observed_energy_distance": observed,
        "exact_permutation_p": exceed / permutation_count,
        "permutation_count": permutation_count,
    }


def concatenate_histograms(
    vectors: dict[str, np.ndarray], channels: tuple[str, ...]
) -> np.ndarray:
    """拼接多个通道的直方图，并让每个通道获得相同总权重。"""

    return np.concatenate([vectors[channel] / len(channels) for channel in channels])


def compute_distances_to_train(
    feature_rows: list[dict[str, Any]],
    histogram_vectors: dict[str, dict[str, np.ndarray]],
) -> list[dict[str, Any]]:
    """计算每张图到训练类 20 颜色分布中心的 Jensen-Shannon 距离。

    训练图使用 leave-one-out 中心，避免其距离因参与中心估计而被系统性压低。
    """

    train_stems = [row["source_stem"] for row in feature_rows if row["group"] == "train_20"]
    rows: list[dict[str, Any]] = []
    color_spaces = {
        "HSV": ("hue_deg", "saturation", "value"),
        "Lab": ("lab_l", "lab_a", "lab_b"),
    }
    for image in feature_rows:
        stem = image["source_stem"]
        reference_stems = [candidate for candidate in train_stems if candidate != stem]
        for color_space, channels in color_spaces.items():
            vector = concatenate_histograms(histogram_vectors[stem], channels)
            centroid = np.mean(
                [concatenate_histograms(histogram_vectors[item], channels) for item in reference_stems],
                axis=0,
            )
            rows.append(
                {
                    "group": image["group"],
                    "group_label": image["group_label"],
                    "split": image["split"],
                    "source_stem": stem,
                    "color_space": color_space,
                    "jensen_shannon_distance": float(jensenshannon(vector, centroid, base=2.0)),
                    "reference_train_image_count": len(reference_stems),
                    "train_reference": "leave-one-out" if image["group"] == "train_20" else "all_train",
                }
            )
    return rows


def distance_statistic_from_partition(
    vectors: np.ndarray,
    target_mask: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray]:
    """根据候选分组重算中心，返回目标距离减训练 LOO 距离。

    每次置换都重新定义训练中心，保证该距离统计量在标签置换下是可交换的。
    """

    train_mask = ~target_mask
    train_vectors = vectors[train_mask]
    target_vectors = vectors[target_mask]
    train_centroid = np.mean(train_vectors, axis=0)
    target_distances = np.asarray(
        [jensenshannon(vector, train_centroid, base=2.0) for vector in target_vectors],
        dtype=np.float64,
    )
    train_distances = []
    for index, vector in enumerate(train_vectors):
        loo_centroid = np.mean(np.delete(train_vectors, index, axis=0), axis=0)
        train_distances.append(jensenshannon(vector, loo_centroid, base=2.0))
    train_distances_array = np.asarray(train_distances, dtype=np.float64)
    statistic = float(np.mean(target_distances) - np.mean(train_distances_array))
    return statistic, target_distances, train_distances_array


def compute_distance_tests(
    feature_rows: list[dict[str, Any]],
    histogram_vectors: dict[str, dict[str, np.ndarray]],
) -> list[dict[str, Any]]:
    """以每次置换重算中心的方式检验目标组是否远离训练分布。"""

    selected = [row for row in feature_rows if row["group"] in {"target_20_4", "train_20"}]
    target_mask = np.asarray([row["group"] == "target_20_4" for row in selected])
    target_n = int(np.sum(target_mask))
    color_spaces = {
        "HSV": ("hue_deg", "saturation", "value"),
        "Lab": ("lab_l", "lab_a", "lab_b"),
    }
    tests: list[dict[str, Any]] = []
    for color_space, channels in color_spaces.items():
        vectors = np.asarray(
            [
                concatenate_histograms(histogram_vectors[row["source_stem"]], channels)
                for row in selected
            ],
            dtype=np.float64,
        )
        observed, target_distances, train_distances = distance_statistic_from_partition(
            vectors, target_mask
        )
        exceed = 0
        permutation_count = 0
        for chosen in itertools.combinations(range(len(selected)), target_n):
            candidate_mask = np.zeros(len(selected), dtype=bool)
            candidate_mask[list(chosen)] = True
            candidate, _, _ = distance_statistic_from_partition(vectors, candidate_mask)
            # 事先定义的单侧假设：目标来源组比训练类20更远。
            exceed += int(candidate >= observed - 1e-12)
            permutation_count += 1
        tests.append(
            {
                "color_space": color_space,
                "target_mean_distance": float(np.mean(target_distances)),
                "train_mean_distance": float(np.mean(train_distances)),
                "target_minus_train": observed,
                "hedges_g": hedges_g(target_distances, train_distances),
                "cliffs_delta": cliffs_delta(target_distances, train_distances),
                "exact_permutation_p": exceed / permutation_count,
                "permutation_alternative": "target_farther_than_train",
                "permutation_recomputes_centroid": True,
                "permutation_count": permutation_count,
            }
        )
    q_values = benjamini_hochberg([row["exact_permutation_p"] for row in tests])
    for row, q_value in zip(tests, q_values):
        row["bh_q"] = q_value
    return tests


def compute_holdout_control_tests(
    distance_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """比较目标组与同属 val/test 的其他类别20图，排除单纯的拆分效应。"""

    tests: list[dict[str, Any]] = []
    for color_space in ("HSV", "Lab"):
        target = np.asarray(
            [
                row["jensen_shannon_distance"]
                for row in distance_rows
                if row["group"] == "target_20_4" and row["color_space"] == color_space
            ],
            dtype=np.float64,
        )
        other = np.asarray(
            [
                row["jensen_shannon_distance"]
                for row in distance_rows
                if row["group"] == "other_holdout_20" and row["color_space"] == color_space
            ],
            dtype=np.float64,
        )
        statistic = lambda a, b: float(np.mean(a) - np.mean(b))
        pooled = np.concatenate([target, other])
        observed = statistic(target, other)
        exceed = 0
        permutation_count = 0
        for chosen in itertools.combinations(range(len(pooled)), len(target)):
            mask = np.zeros(len(pooled), dtype=bool)
            mask[list(chosen)] = True
            exceed += int(statistic(pooled[mask], pooled[~mask]) >= observed - 1e-12)
            permutation_count += 1
        tests.append(
            {
                "color_space": color_space,
                "target_n": len(target),
                "other_holdout_n": len(other),
                "target_mean_distance": float(np.mean(target)),
                "other_holdout_mean_distance": float(np.mean(other)),
                "target_minus_other_holdout": observed,
                "hedges_g": hedges_g(target, other),
                "cliffs_delta": cliffs_delta(target, other),
                "exact_permutation_p": exceed / permutation_count,
                "permutation_alternative": "target_farther_than_other_holdout",
                "permutation_count": permutation_count,
            }
        )
    q_values = benjamini_hochberg([row["exact_permutation_p"] for row in tests])
    for row, q_value in zip(tests, q_values):
        row["bh_q"] = q_value
    return tests


def configure_matplotlib() -> None:
    """设置适合论文双栏图的字体和矢量导出参数。"""

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7,
            "axes.labelsize": 7,
            "axes.titlesize": 8,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.7,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "legend.fontsize": 6.5,
            "legend.frameon": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def save_figure(fig: plt.Figure, base_path: Path, dpi: int) -> None:
    """统一导出可编辑矢量图和高分辨率预览图。"""

    fig.savefig(base_path.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(base_path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base_path.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_distributions(
    histogram_rows: list[dict[str, Any]],
    config: HSVLabAnalysisConfig,
) -> None:
    """绘制六个颜色通道的逐图平均分布与图像间 95% 区间。"""

    configure_matplotlib()
    width = config.figure_width_mm / 25.4
    height = config.distribution_height_mm / 25.4
    fig, axes = plt.subplots(2, 3, figsize=(width, height), constrained_layout=True)

    channel_order = ("hue_deg", "saturation", "value", "lab_l", "lab_a", "lab_b")
    panel_labels = "abcdef"
    for axis, channel, panel_label in zip(axes.flat, channel_order, panel_labels):
        channel_rows = [row for row in histogram_rows if row["channel"] == channel]
        for group in GROUP_ORDER:
            group_rows = [row for row in channel_rows if row["group"] == group]
            stems = sorted({row["source_stem"] for row in group_rows})
            matrix = np.asarray(
                [
                    [
                        row["probability"]
                        for row in group_rows
                        if row["source_stem"] == stem
                    ]
                    for stem in stems
                ],
                dtype=np.float64,
            )
            centers = np.asarray(
                [
                    row["bin_center"]
                    for row in group_rows
                    if row["source_stem"] == stems[0]
                ],
                dtype=np.float64,
            )
            mean = np.mean(matrix, axis=0)
            low, high = np.quantile(matrix, [0.025, 0.975], axis=0)
            color = GROUP_COLORS[group]
            axis.plot(centers, mean, color=color, linewidth=1.5, label=f"{GROUP_LABELS[group]} (n={len(stems)})")
            axis.fill_between(centers, low, high, color=color, alpha=0.12, linewidth=0)
        label = HISTOGRAM_SPECS[channel][0]
        axis.set_xlabel(label)
        axis.set_ylabel("Pixel probability")
        axis.text(-0.16, 1.06, panel_label, transform=axis.transAxes, fontweight="bold", fontsize=8)
        axis.grid(axis="y", color="#E5E7EB", linewidth=0.5)
    axes[0, 0].legend(loc="upper right")
    fig.suptitle("Original-image HSV and CIELAB distributions", fontsize=9, fontweight="bold")
    save_figure(fig, config.output_dir / "hsv_lab_distributions", config.preview_dpi)


def plot_shift_summary(
    feature_rows: list[dict[str, Any]],
    distance_rows: list[dict[str, Any]],
    feature_tests: list[dict[str, Any]],
    config: HSVLabAnalysisConfig,
) -> None:
    """绘制 Lab 色度、训练中心距离、逐图特征和效应量摘要。"""

    configure_matplotlib()
    rng = np.random.default_rng(config.random_seed)
    width = config.figure_width_mm / 25.4
    height = config.summary_height_mm / 25.4
    fig = plt.figure(figsize=(width, height), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, height_ratios=(1.0, 1.12))
    ax_ab = fig.add_subplot(grid[0, 0])
    ax_distance = fig.add_subplot(grid[0, 1])
    ax_heatmap = fig.add_subplot(grid[1, 0])
    ax_effect = fig.add_subplot(grid[1, 1])

    # a：Lab a*/b* 色度平面，每个点是一张原图。
    markers = {"train_20": "o", "target_20_4": "D", "other_holdout_20": "^"}
    for group in GROUP_ORDER:
        subset = [row for row in feature_rows if row["group"] == group]
        ax_ab.scatter(
            [row["lab_a_mean"] for row in subset],
            [row["lab_b_mean"] for row in subset],
            s=24 if group == "target_20_4" else 18,
            marker=markers[group],
            color=GROUP_COLORS[group],
            edgecolor="white",
            linewidth=0.5,
            label=f"{GROUP_LABELS[group]} (n={len(subset)})",
            zorder=3,
        )
        ax_ab.scatter(
            np.mean([row["lab_a_mean"] for row in subset]),
            np.mean([row["lab_b_mean"] for row in subset]),
            s=70,
            marker="X",
            color=GROUP_COLORS[group],
            edgecolor="#111827",
            linewidth=0.6,
            zorder=4,
        )
    ax_ab.set_xlabel("Mean a*")
    ax_ab.set_ylabel("Mean b*")
    ax_ab.set_title("CIELAB chromaticity per image")
    ax_ab.grid(color="#E5E7EB", linewidth=0.5)
    ax_ab.legend(loc="best")

    # b：到训练类 20 分布中心的距离；训练图采用 leave-one-out 中心。
    x_positions = {("HSV", group): index for index, group in enumerate(GROUP_ORDER)}
    x_positions.update({("Lab", group): index + 4 for index, group in enumerate(GROUP_ORDER)})
    for row in distance_rows:
        x = x_positions[(row["color_space"], row["group"])] + rng.uniform(-0.10, 0.10)
        ax_distance.scatter(
            x,
            row["jensen_shannon_distance"],
            s=18,
            color=GROUP_COLORS[row["group"]],
            marker=markers[row["group"]],
            edgecolor="white",
            linewidth=0.4,
            zorder=3,
        )
    for color_space in ("HSV", "Lab"):
        for group in GROUP_ORDER:
            subset = [
                row["jensen_shannon_distance"]
                for row in distance_rows
                if row["color_space"] == color_space and row["group"] == group
            ]
            x = x_positions[(color_space, group)]
            ax_distance.plot([x - 0.16, x + 0.16], [np.mean(subset), np.mean(subset)], color="#111827", linewidth=1.2)
    ax_distance.axvline(3.0, color="#CBD5E1", linewidth=0.8)
    ax_distance.set_xticks(
        [0, 1, 2, 4, 5, 6],
        ["Train", "20/4-*", "Other", "Train", "20/4-*", "Other"],
        rotation=25,
        ha="right",
    )
    # 色彩空间标签放在坐标区内部，避免与面板标题重叠。
    ax_distance.text(1, 0.98, "HSV", transform=ax_distance.get_xaxis_transform(), ha="center", va="top", fontweight="bold")
    ax_distance.text(5, 0.98, "Lab", transform=ax_distance.get_xaxis_transform(), ha="center", va="top", fontweight="bold")
    ax_distance.set_ylabel("Jensen–Shannon distance to train")
    ax_distance.set_title("Distance from the training color distribution")
    ax_distance.grid(axis="y", color="#E5E7EB", linewidth=0.5)

    # c：逐图标准化特征热图，展示偏移是否由少数异常图驱动。
    heat_features = [
        ("saturation_mean", "S"),
        ("value_mean", "V"),
        ("lab_l_mean", "L*"),
        ("lab_a_mean", "a*"),
        ("lab_b_mean", "b*"),
        ("lab_chroma_mean", "C*"),
    ]
    ordered_rows = [row for group in GROUP_ORDER for row in feature_rows if row["group"] == group]
    raw = np.asarray([[row[key] for key, _ in heat_features] for row in ordered_rows], dtype=np.float64)
    train_raw = np.asarray(
        [[row[key] for key, _ in heat_features] for row in feature_rows if row["group"] == "train_20"],
        dtype=np.float64,
    )
    train_mean = np.mean(train_raw, axis=0)
    train_std = np.std(train_raw, axis=0, ddof=1)
    train_std[train_std == 0] = 1.0
    z = (raw - train_mean) / train_std
    image = ax_heatmap.imshow(z.T, aspect="auto", cmap="RdBu_r", vmin=-3.0, vmax=3.0, interpolation="nearest")
    ax_heatmap.set_yticks(range(len(heat_features)), [label for _, label in heat_features])
    ax_heatmap.set_xticks(range(len(ordered_rows)), [row["source_stem"] for row in ordered_rows], rotation=90)
    train_count = sum(row["group"] == "train_20" for row in ordered_rows)
    target_count = sum(row["group"] == "target_20_4" for row in ordered_rows)
    ax_heatmap.axvline(train_count - 0.5, color="white", linewidth=1.5)
    ax_heatmap.axvline(train_count + target_count - 0.5, color="white", linewidth=1.5)
    ax_heatmap.text((train_count - 1) / 2, -0.72, "Train", ha="center", fontsize=6.5)
    ax_heatmap.text(train_count + (target_count - 1) / 2, -0.72, "20/4-*", ha="center", fontsize=6.5, color=GROUP_COLORS["target_20_4"])
    ax_heatmap.text(train_count + target_count + (len(ordered_rows) - train_count - target_count - 1) / 2, -0.72, "Other held-out", ha="center", fontsize=6.5)
    ax_heatmap.set_title("Per-image features standardized to training class 20", pad=25)
    colorbar = fig.colorbar(image, ax=ax_heatmap, fraction=0.04, pad=0.02)
    colorbar.set_label("Training z-score")

    # d：非圆周特征的 Hedges' g；旁边直接标出 BH 校正 q 值。
    effect_rows = [row for row in feature_tests if not row["circular_feature"]]
    y = np.arange(len(effect_rows))
    effects = np.asarray([row["hedges_g"] for row in effect_rows])
    colors = ["#D95F59" if effect > 0 else "#3A8D8A" for effect in effects]
    ax_effect.axvline(0.0, color="#64748B", linewidth=0.8)
    ax_effect.barh(y, effects, color=colors, alpha=0.86, height=0.65)
    ax_effect.set_yticks(y, [row["feature_label"] for row in effect_rows])
    ax_effect.invert_yaxis()
    ax_effect.set_xlabel("Hedges' g (20/4-* minus train)")
    ax_effect.set_title("Image-level effect sizes and exact tests")
    ax_effect.grid(axis="x", color="#E5E7EB", linewidth=0.5)
    x_span = max(1.0, float(np.max(np.abs(effects))) * 1.35)
    ax_effect.set_xlim(-x_span, x_span)
    for index, row in enumerate(effect_rows):
        anchor = x_span * 0.97
        ax_effect.text(anchor, index, f"q={row['bh_q']:.3g}", ha="right", va="center", fontsize=6.2)

    for panel_label, axis in zip("abcd", (ax_ab, ax_distance, ax_heatmap, ax_effect)):
        axis.text(-0.14, 1.07, panel_label, transform=axis.transAxes, fontweight="bold", fontsize=8)

    fig.suptitle("Is 20/4-* color-shifted relative to training class 20?", fontsize=9, fontweight="bold")
    save_figure(fig, config.output_dir / "hsv_lab_shift_summary", config.preview_dpi)


def format_float(value: float, digits: int = 4) -> str:
    """把浮点数格式化为报告文本。"""

    if not math.isfinite(value):
        return "NA"
    return f"{value:.{digits}f}"


def build_report(
    config: HSVLabAnalysisConfig,
    feature_rows: list[dict[str, Any]],
    feature_tests: list[dict[str, Any]],
    multivariate_test: dict[str, Any],
    distance_tests: list[dict[str, Any]],
    holdout_control_tests: list[dict[str, Any]],
) -> str:
    """生成可直接阅读的中文分析报告。"""

    group_counts = {
        group: sum(row["group"] == group for row in feature_rows) for group in GROUP_ORDER
    }
    significant_features = [row for row in feature_tests if row["bh_q"] < 0.05]
    multivariate_significant = multivariate_test["exact_permutation_p"] < 0.05
    distance_significant = [row for row in distance_tests if row["bh_q"] < 0.05]

    if multivariate_significant and (significant_features or distance_significant):
        verdict = "存在图像级证据支持 20/4-* 相对训练类 20 的真实颜色分布偏移。"
    elif multivariate_significant or significant_features or distance_significant:
        verdict = "检测到部分颜色偏移信号，但单变量与多变量证据尚不完全一致，应视为有限证据。"
    else:
        verdict = "目前没有图像级统计证据证明 20/4-* 超出训练类 20 的自然颜色波动。"

    target_rows = sorted(
        [row for row in feature_rows if row["group"] == "target_20_4"],
        key=lambda row: row["source_stem"],
    )
    target_stems = [row["source_stem"] for row in target_rows]
    target_splits = [f"{row['source_stem']}={row['split']}" for row in target_rows]
    report = [
        "# 类别 20 的 4-* 原图 HSV/Lab 偏移分析",
        "",
        "## 先说明 4-* 是什么",
        "",
        f"- `4-*` 是文件名通配写法，具体就是：{', '.join(f'`{item}`' for item in target_stems)}。",
        "- `4` 表示文件命名中的来源组键；在没有采集记录佐证前，不能擅自把它解释为“第4批”“第4台设备”或某个具体处理。",
        "- `*` 表示该来源组内的编号 1～4。它不是乘法，也不是一个单独样本。",
        f"- 当前拆分位置：{', '.join(f'`{item}`' for item in target_splits)}；因此训练类 20 完全没有见过 `4-*`。",
        "",
        "## 核心结论",
        "",
        f"**{verdict}**",
        "",
        f"- 多元 HSV/Lab 能量距离精确置换检验：p={multivariate_test['exact_permutation_p']:.6f}，穷举 {multivariate_test['permutation_count']} 种同样本量分组。",
        f"- 独立样本数：训练类20 n={group_counts['train_20']}，20/4-* n={group_counts['target_20_4']}，类别20其他留出图 n={group_counts['other_holdout_20']}。",
        "- 像素只用于估计每张图的分布；统计检验的 n 是原图数，不是像素数。",
        "",
        "## 单变量图像级检验",
        "",
        "| 特征 | 20/4-*中心 | 训练中心 | 差值 | 95% bootstrap CI | Hedges g | 精确p | BH q |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in feature_tests:
        report.append(
            "| {feature_label} | {target} | {train} | {difference} | [{low}, {high}] | {g} | {p:.6f} | {q:.6f} |".format(
                feature_label=row["feature_label"],
                target=format_float(row["target_center"]),
                train=format_float(row["train_center"]),
                difference=format_float(row["target_minus_train"]),
                low=format_float(row["bootstrap_ci95_low"]),
                high=format_float(row["bootstrap_ci95_high"]),
                g=format_float(row["hedges_g"]),
                p=row["exact_permutation_p"],
                q=row["bh_q"],
            )
        )

    report.extend(
        [
            "",
            "Hue 为圆周变量，其差值是最短圆周角差，因此不报告线性 Hedges g。其余正差值表示 20/4-* 更高。",
            "",
            "## 到训练颜色分布中心的距离",
            "",
            "| 色彩空间 | 20/4-*平均距离 | 训练LOO平均距离 | 差值 | Hedges g | 精确p | BH q |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in distance_tests:
        report.append(
            "| {color_space} | {target_mean_distance:.5f} | {train_mean_distance:.5f} | {target_minus_train:.5f} | {hedges_g:.4f} | {exact_permutation_p:.6f} | {bh_q:.6f} |".format(
                **row
            )
        )
    report.extend(
        [
            "",
            "距离置换检验采用预先定义的单侧假设（20/4-* 更远），并在每一种候选分组中重新计算训练中心和训练 leave-one-out 距离。",
            "",
            "## 与类别20其他留出图的拆分对照",
            "",
            "| 色彩空间 | 20/4-*平均距离 | 其他留出图平均距离 | 差值 | Hedges g | 精确单侧p | BH q |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in holdout_control_tests:
        report.append(
            "| {color_space} | {target_mean_distance:.5f} | {other_holdout_mean_distance:.5f} | {target_minus_other_holdout:.5f} | {hedges_g:.4f} | {exact_permutation_p:.6f} | {bh_q:.6f} |".format(
                **row
            )
        )
    report.extend(
        [
            "",
            "该对照的两组都来自 val/test，且都使用同一个训练类20中心；若 20/4-* 仍显著更远，就不能只用“留出集整体与训练集不同”解释。",
            "",
            "## 方法与限制",
            "",
            f"- 直接读取原始 BMP；不进行增强、白平衡、亮度、对比度或 gamma 调整。",
            f"- 为控制计算量，从原图每隔 {config.pixel_stride} 个像素固定取样一次；每张图仍有约 31 万个像素用于估计分布。",
            "- Hue 使用饱和度加权圆周统计，避免灰暗像素的色相噪声主导结果。",
            "- 单变量 p 值由图像级精确置换检验获得，并以 BH 法控制多重比较；区间为图像级 bootstrap 95% CI。",
            "- 训练图到训练中心的距离采用 leave-one-out；否则训练图因参与中心估计会获得不公平的小距离。",
            "- `20/4-*` 只有4张，而且4张图共享同一个文件来源组键 `4`，它们未必是4个独立采集来源；精确置换 p 值依赖原图可交换性，因此应把很小的 p 值理解为当前图像集合内的系统分离证据，而不是跨来源总体推断。",
            "- 显著结果支持“来源组相关颜色偏移”，但不能单凭文件名断言其物理原因。",
            "- 颜色偏移即使存在，也只能说明模型可能利用了颜色线索；仍需结合多seed分类结果与其他来源组控制判断因果。",
            "",
            "## 输出文件",
            "",
            "- `image_features.csv`：每张原图的图像级 HSV/Lab 特征。",
            "- `pixel_histograms.csv`：六个颜色通道的逐图概率分布。",
            "- `feature_tests.csv`：单变量效应量、精确 p 值与 BH q 值。",
            "- `distance_to_train.csv` / `distance_tests.csv`：到训练颜色中心的距离及重算中心置换检验。",
            "- `holdout_control_tests.csv`：20/4-* 对类别20其他留出图的拆分对照。",
            "- `hsv_lab_distributions.*`：HSV/Lab 分布图。",
            "- `hsv_lab_shift_summary.*`：图像级偏移证据摘要图。",
            "",
        ]
    )
    return "\n".join(report)


def validate_config(config: HSVLabAnalysisConfig) -> None:
    """运行前检查配置，避免静默使用错误路径或样本。"""

    if not config.manifest_path.is_file():
        raise FileNotFoundError(f"找不到拆分清单：{config.manifest_path}")
    if config.pixel_stride <= 0:
        raise ValueError("pixel_stride 必须是正整数。")
    if config.bootstrap_iterations < 1000:
        raise ValueError("bootstrap_iterations 至少应为 1000。")


def main() -> None:
    """执行完整 HSV/Lab 分析。"""

    config = CONFIG
    validate_config(config)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    manifest = read_manifest(config.manifest_path)
    selected = [(row, classify_group(row, config)) for row in manifest]
    selected = [(row, group) for row, group in selected if group is not None]

    feature_rows: list[dict[str, Any]] = []
    histogram_rows: list[dict[str, Any]] = []
    histogram_vectors: dict[str, dict[str, np.ndarray]] = {}
    for index, (row, group) in enumerate(selected, start=1):
        path = Path(row["source_path"])
        print(f"[{index:02d}/{len(selected):02d}] 计算 {row['time_code']}/{row['source_stem']} ({row['split']})")
        arrays = load_color_arrays(path, config.pixel_stride)
        image_features = summarize_image(row, group, arrays)
        image_histograms, vectors = build_histogram_rows(image_features, arrays)
        feature_rows.append(image_features)
        histogram_rows.extend(image_histograms)
        histogram_vectors[row["source_stem"]] = vectors

    counts = {group: sum(row["group"] == group for row in feature_rows) for group in GROUP_ORDER}
    expected_target = {"4-1", "4-2", "4-3", "4-4"}
    actual_target = {row["source_stem"] for row in feature_rows if row["group"] == "target_20_4"}
    if actual_target != expected_target:
        raise ValueError(f"20/4-* 样本不符合预期：{sorted(actual_target)}")
    if counts != {"train_20": 15, "target_20_4": 4, "other_holdout_20": 5}:
        raise ValueError(f"类别20分组数量不符合预期：{counts}")

    feature_tests = compute_feature_tests(feature_rows, config)
    multivariate_test = compute_multivariate_test(feature_rows)
    distance_rows = compute_distances_to_train(feature_rows, histogram_vectors)
    distance_tests = compute_distance_tests(feature_rows, histogram_vectors)
    holdout_control_tests = compute_holdout_control_tests(distance_rows)

    write_csv(config.output_dir / "image_features.csv", feature_rows)
    write_csv(config.output_dir / "pixel_histograms.csv", histogram_rows)
    write_csv(config.output_dir / "feature_tests.csv", feature_tests)
    write_csv(config.output_dir / "distance_to_train.csv", distance_rows)
    write_csv(config.output_dir / "distance_tests.csv", distance_tests)
    write_csv(config.output_dir / "holdout_control_tests.csv", holdout_control_tests)
    write_csv(config.output_dir / "multivariate_test.csv", [multivariate_test])

    summary = {
        "config": {key: str(value) if isinstance(value, Path) else value for key, value in asdict(config).items()},
        "group_counts": counts,
        "target_files": sorted(actual_target),
        "target_split_positions": {
            row["source_stem"]: row["split"] for row in feature_rows if row["group"] == "target_20_4"
        },
        "feature_tests": feature_tests,
        "multivariate_test": multivariate_test,
        "distance_tests": distance_tests,
        "holdout_control_tests": holdout_control_tests,
    }
    (config.output_dir / "analysis_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    plot_distributions(histogram_rows, config)
    plot_shift_summary(feature_rows, distance_rows, feature_tests, config)
    report = build_report(
        config,
        feature_rows,
        feature_tests,
        multivariate_test,
        distance_tests,
        holdout_control_tests,
    )
    report_path = config.output_dir / "HSV_LAB_REPORT.md"
    report_path.write_text(report, encoding="utf-8")

    print("\nHSV/Lab 原图分析完成。")
    print(f"报告：{report_path}")
    print(f"多元精确置换 p={multivariate_test['exact_permutation_p']:.6f}")


if __name__ == "__main__":
    main()
