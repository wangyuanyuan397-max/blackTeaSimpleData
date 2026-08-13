"""诊断工具的轻量单元测试，不依赖训练和网络下载。"""

from __future__ import annotations

import random
import sys
import unittest
from pathlib import Path

import numpy as np
from PIL import Image


EXPERIMENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EXPERIMENT_DIR))

from diagnosis_common import (  # noqa: E402
    PerturbationSpec,
    adjust_white_balance_temperature,
    apply_perturbation,
    classification_metrics,
    infer_parent_id,
    shuffle_tiles,
)


class DiagnosisCommonTests(unittest.TestCase):
    def test_parent_id_can_restore_random_and_grid_patch(self) -> None:
        """两种现有派生 patch 命名都应恢复到同一原图层级。"""

        self.assertEqual(
            infer_parent_id(Path("t00__1-4__random55_001.jpg"), "00"),
            "00/1-4",
        )
        self.assertEqual(
            infer_parent_id(Path("t20__4-2__grid30_30.jpg"), "20"),
            "20/4-2",
        )

    def test_occlusion_is_deterministic_and_changes_pixels(self) -> None:
        """相同样本键必须得到完全相同的遮挡。"""

        array = np.arange(120 * 100 * 3, dtype=np.uint8).reshape(100, 120, 3)
        image = Image.fromarray(array)
        spec = PerturbationSpec("occlusion", 0.3)
        first = np.asarray(apply_perturbation(image, spec, "sample", 2026))
        second = np.asarray(apply_perturbation(image, spec, "sample", 2026))
        self.assertTrue(np.array_equal(first, second))
        self.assertFalse(np.array_equal(first, array))

    def test_patch_shuffle_preserves_pixel_multiset(self) -> None:
        """可整除图像的 patch shuffle 只能重排像素，不能改变像素集合。"""

        array = np.arange(12 * 12 * 3, dtype=np.uint8).reshape(12, 12, 3)
        shuffled = np.asarray(shuffle_tiles(Image.fromarray(array), 3, random.Random(7)))
        original_pixels = sorted(map(tuple, array.reshape(-1, 3).tolist()))
        shuffled_pixels = sorted(map(tuple, shuffled.reshape(-1, 3).tolist()))
        self.assertEqual(original_pixels, shuffled_pixels)

    def test_adjacent_error_fraction_uses_only_errors_as_denominator(self) -> None:
        """相邻错误占比的分母应是错误样本数，而非全部样本数。"""

        labels = np.asarray([0, 1, 2, 3, 4])
        predictions = [0, 2, 4, 3, 1]
        probabilities = np.full((5, 5), 0.01)
        for row, prediction in enumerate(predictions):
            probabilities[row, prediction] = 0.96
        metrics = classification_metrics(labels, probabilities, ["00", "10", "20", "30", "40"])
        # 错误共 3 个，其中只有 1->2 是相邻错误。
        self.assertAlmostEqual(metrics["adjacent_error_fraction"], 1 / 3)
        self.assertAlmostEqual(metrics["far_error_fraction"], 2 / 3)

    def test_color_components_are_independent_and_deterministic(self) -> None:
        """拆分颜色算子应可复现，并且不同分量产生不同像素结果。"""

        yy, xx = np.mgrid[0:32, 0:40]
        array = np.stack(
            [
                20 + xx * 4,
                30 + yy * 5,
                40 + (xx + yy) * 2,
            ],
            axis=2,
        ).clip(0, 255).astype(np.uint8)
        image = Image.fromarray(array)
        specs = [
            PerturbationSpec("brightness", 0.8),
            PerturbationSpec("contrast", 0.8),
            PerturbationSpec("saturation", 0.7),
            PerturbationSpec("hue", 0.05),
            PerturbationSpec("white_balance_temperature", 0.15),
        ]
        outputs = []
        for spec in specs:
            first = np.asarray(apply_perturbation(image, spec, "sample", 2026))
            second = np.asarray(apply_perturbation(image, spec, "sample", 2026))
            self.assertTrue(np.array_equal(first, second))
            self.assertFalse(np.array_equal(first, array))
            outputs.append(first)
        self.assertEqual(len({output.tobytes() for output in outputs}), len(outputs))

    def test_temperature_direction_and_mean_luminance(self) -> None:
        """暖色应提高红蓝比，同时不应明显改变全图平均亮度。"""

        image = Image.fromarray(np.full((40, 50, 3), [110, 130, 150], dtype=np.uint8))
        warm = np.asarray(adjust_white_balance_temperature(image, 0.15), dtype=np.float32)
        cool = np.asarray(adjust_white_balance_temperature(image, -0.15), dtype=np.float32)
        self.assertGreater(warm[..., 0].mean() / warm[..., 2].mean(), 110 / 150)
        self.assertLess(cool[..., 0].mean() / cool[..., 2].mean(), 110 / 150)
        weights = np.asarray([0.2126, 0.7152, 0.0722], dtype=np.float32)
        original_luminance = (np.asarray(image, dtype=np.float32) * weights).sum(2).mean()
        warm_luminance = (warm * weights).sum(2).mean()
        cool_luminance = (cool * weights).sum(2).mean()
        self.assertLess(abs(warm_luminance - original_luminance), 1.0)
        self.assertLess(abs(cool_luminance - original_luminance), 1.0)


if __name__ == "__main__":
    unittest.main()
