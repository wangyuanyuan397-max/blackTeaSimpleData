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


if __name__ == "__main__":
    unittest.main()
