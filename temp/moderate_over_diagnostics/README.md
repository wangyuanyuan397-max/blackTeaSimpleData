# Moderate vs Over Diagnostics

这套临时脚本用于诊断 `Moderate` 和 `Over` 为什么难分，不修改主训练框架。

数据来源：

- `datas_test_point`: 按时间点保存的原始大图。
- `datas_test_point_30_patches`: 每张原图裁成 30 个 patch 后的数据。

标签定义：

- `30, 35, 40, 45` -> `moderate`，标签 0。
- `50, 55, 60` -> `over`，标签 1。

一键运行：

```bash
cd E:\workspaces\python\BlackTeaSimpleData\temp\moderate_over_diagnostics
python run_all.py --epochs 30 --device auto
```

快速测试：

```bash
python run_all.py --epochs 1 --patience 1 --device cpu
```

分步运行：

```bash
python build_metadata.py --run-dir outputs/debug
python split_sources.py --run-dir outputs/debug --seed 2026
python train_binary_cnn.py --run-dir outputs/debug --dataset original --variant rgb --epochs 30
python train_binary_cnn.py --run-dir outputs/debug --dataset patch --variant rgb --epochs 30
python train_binary_cnn.py --run-dir outputs/debug --dataset patch --variant gray --epochs 30
python train_binary_cnn.py --run-dir outputs/debug --dataset patch --variant blur --epochs 30
python diagnose_patch_consistency.py --run-dir outputs/debug
python diagnose_color_texture.py --run-dir outputs/debug
```

主要输出：

- `metadata/source_metadata.csv`: 原图元数据。
- `metadata/patch_metadata.csv`: patch 元数据。
- `splits/source_splits.csv`: 原图级 7:1:2 划分，保证同一原图的 30 个 patch 不跨 split。
- `cnn_original_rgb/metrics.json`: 原图缩放后二分类结果。
- `cnn_patch_rgb/metrics.json`: patch 二分类结果。
- `cnn_patch_gray/metrics.json`: 灰度 patch 二分类结果。
- `cnn_patch_blur/metrics.json`: 模糊 patch 二分类结果。
- `patch_consistency/test_source_patch_consistency.csv`: 每张原图的 30 patch 预测一致性。
- `color_texture/feature_summary.csv`: 颜色/纹理传统特征 SVM 对照。

解释重点：

- 原图明显优于 patch：可能需要整体颜色分布或空间上下文。
- 原图和 patch 都低：Moderate/Over 视觉重叠或标签边界可能本身模糊。
- patch 反而更好：局部细节确实有诊断价值。
- RGB 或颜色直方图明显优于灰度：颜色是主要判别依据。
- 灰度接近 RGB：纹理或形态信息也有较强判别力。
