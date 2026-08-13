"""诊断闭环新增统计与选择规则的轻量测试。"""

from __future__ import annotations

import re
import unittest

from audit_source_groups import add_bh_q_values, fisher_two_sided, parse_source_stem
from select_fusion_on_validation import candidate_key, paired_correctness_comparison
from run_color_component_experiments import (
    ColorComponentConfig,
    add_bh_q_values as add_color_bh_q_values,
    build_conditions as build_color_conditions,
    conditions_for_split as color_conditions_for_split,
    condition_name as color_condition_name,
    validate_config as validate_color_config,
)


class DiagnosticClosureTests(unittest.TestCase):
    def test_parse_source_stem(self) -> None:
        """文件名两段编号应分别解析为来源组和组内重复。"""

        pattern = re.compile(r"^(\d+)-(\d+)$")
        self.assertEqual(parse_source_stem("4-2", pattern), ("4", "2"))
        with self.assertRaises(ValueError):
            parse_source_stem("4", pattern)

    def test_fisher_two_sided_known_table(self) -> None:
        """无 scipy 实现应与经典 Fisher 双侧示例一致。"""

        self.assertAlmostEqual(fisher_two_sided(1, 9, 11, 3), 0.002759456, places=8)

    def test_bh_q_values_are_monotonic_after_sorting(self) -> None:
        """BH 校正后的 q 值应正确回填到原始行。"""

        rows = [
            {"fisher_two_sided_p": 0.01},
            {"fisher_two_sided_p": 0.04},
            {"fisher_two_sided_p": 0.03},
        ]
        add_bh_q_values(rows)
        actual = [row["bh_fdr_q"] for row in rows]
        expected = [0.03, 0.04, 0.04]
        for left, right in zip(actual, expected):
            self.assertAlmostEqual(left, right)

    def test_candidate_key_prefers_fewer_components_on_complete_tie(self) -> None:
        """所有验证指标相同时，应选择更克制的单尺度。"""

        metrics = {"accuracy": 0.8, "qwk": 0.7, "macro_f1": 0.75, "nll": 0.5}
        single = {"component_count": 1, "metrics": metrics}
        fusion = {"component_count": 2, "metrics": metrics}
        self.assertGreater(candidate_key(single, "accuracy"), candidate_key(fusion, "accuracy"))

    def test_paired_correctness_comparison(self) -> None:
        """配对统计应同时记录融合新增答对和损失答对。"""

        baseline = [
            {"parent_id": "a", "correct": 1},
            {"parent_id": "b", "correct": 1},
            {"parent_id": "c", "correct": 0},
        ]
        candidate = [
            {"parent_id": "a", "correct": 1},
            {"parent_id": "b", "correct": 0},
            {"parent_id": "c", "correct": 1},
        ]
        result = paired_correctness_comparison(baseline, candidate)
        self.assertEqual(result["fusion_gained_correct"], 1)
        self.assertEqual(result["fusion_lost_correct"], 1)
        self.assertEqual(result["mcnemar_exact_two_sided_p"], 1.0)

    def test_color_conditions_cover_five_independent_components(self) -> None:
        """默认颜色实验应包含原图和五类各四档单因素条件。"""

        config = ColorComponentConfig()
        validate_color_config(config)
        conditions = build_color_conditions(config)
        counts: dict[str, int] = {}
        names = []
        for condition in conditions:
            counts[condition.name] = counts.get(condition.name, 0) + 1
            names.append(color_condition_name(condition))
        self.assertEqual(counts["original"], 1)
        for component in (
            "brightness",
            "contrast",
            "saturation",
            "hue",
            "white_balance_temperature",
        ):
            self.assertEqual(counts[component], 4)
        self.assertEqual(len(names), 21)
        self.assertEqual(len(set(names)), len(names))

    def test_color_test_requires_and_obeys_frozen_conditions(self) -> None:
        """测试集必须拒绝全量搜索，只允许明确冻结的颜色条件。"""

        from dataclasses import replace

        unsafe = replace(ColorComponentConfig(), split="test")
        with self.assertRaises(ValueError):
            validate_color_config(unsafe)
        safe = replace(
            unsafe,
            frozen_test_conditions=(("brightness", 0.70), ("hue", -0.06)),
        )
        validate_color_config(safe)
        conditions = color_conditions_for_split(safe)
        self.assertEqual(
            [(condition.name, condition.value) for condition in conditions],
            [("original", 0.0), ("brightness", 0.70), ("hue", -0.06)],
        )

    def test_color_bh_correction_excludes_original(self) -> None:
        """颜色多重校正只应以实际扰动条件为检验家族。"""

        rows = [
            {"perturbation": "original", "paired_mcnemar_exact_p": 1.0},
            {"perturbation": "hue", "paired_mcnemar_exact_p": 0.01},
            {"perturbation": "brightness", "paired_mcnemar_exact_p": 0.04},
            {"perturbation": "contrast", "paired_mcnemar_exact_p": 0.03},
        ]
        add_color_bh_q_values(rows)
        self.assertEqual(rows[0]["paired_mcnemar_bh_q"], 1.0)
        expected = [0.03, 0.04, 0.04]
        for row, value in zip(rows[1:], expected):
            self.assertAlmostEqual(row["paired_mcnemar_bh_q"], value)


if __name__ == "__main__":
    unittest.main()
