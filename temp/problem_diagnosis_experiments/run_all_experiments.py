"""PyCharm 一键运行全部任务诊断实验。

运行顺序：
1. Global/Local 尺度训练（默认不在测试集枚举融合）；
2. 对选定 checkpoint 做受控破坏；
3. 汇总生成中文报告。

各阶段的详细参数分别位于对应脚本顶部，本文件只控制是否执行某阶段。
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path


EXPERIMENT_DIR = Path(__file__).resolve().parent
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))

import run_crop_experiments  # noqa: E402
import run_perturbation_experiments  # noqa: E402
import summarize_diagnosis  # noqa: E402


# =============================================================================
# PyCharm 右键运行配置区
# True 表示执行，False 表示跳过。第一次正式运行建议全部保持 True。
# =============================================================================
RUN_CROP_EXPERIMENTS = True
RUN_PERTURBATION_EXPERIMENTS = True
RUN_SUMMARY_REPORT = True

# 默认受控破坏的是 crop_204。这里根据尺度实验输出目录自动对齐 checkpoint，
# 因此即使在 PyCharm 中修改了 crop 脚本的 output_dir，也不用再同步修改路径。
PERTURBATION_CROP_SIZE: int | str = 204


def main() -> None:
    """按照顶部开关依次执行三个阶段。"""

    if RUN_CROP_EXPERIMENTS:
        print("\n========== 阶段 1/3：Global/Local 尺度诊断 ==========")
        run_crop_experiments.main()
    else:
        print("\n跳过阶段 1：复用现有尺度实验结果。")

    if RUN_PERTURBATION_EXPERIMENTS:
        print("\n========== 阶段 2/3：受控破坏诊断 ==========")
        checkpoint = (
            run_crop_experiments.CONFIG.output_dir
            / f"crop_{PERTURBATION_CROP_SIZE}"
            / "best.pth"
        )
        run_perturbation_experiments.CONFIG = replace(
            run_perturbation_experiments.CONFIG,
            checkpoint=checkpoint,
        )
        run_perturbation_experiments.main()
    else:
        print("\n跳过阶段 2：复用现有扰动实验结果。")

    if RUN_SUMMARY_REPORT:
        print("\n========== 阶段 3/3：生成中文诊断报告 ==========")
        # 报告路径自动跟随前两个阶段的 output_dir。
        summarize_diagnosis.CROP_SUMMARY = (
            run_crop_experiments.CONFIG.output_dir / "summary.csv"
        )
        summarize_diagnosis.PERTURBATION_SUMMARY = (
            run_perturbation_experiments.CONFIG.output_dir / "summary.csv"
        )
        summarize_diagnosis.main()
    else:
        print("\n跳过阶段 3：不生成汇总报告。")

    print("\n全部启用的诊断阶段均已完成。")


if __name__ == "__main__":
    main()
