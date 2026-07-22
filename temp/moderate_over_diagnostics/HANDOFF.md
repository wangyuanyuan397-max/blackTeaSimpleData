# HANDOFF: Moderate vs Over Diagnostics

## 任务目标

本目录是一套临时诊断实验，位置：

```text
E:\workspaces\python\BlackTeaSimpleData\temp\moderate_over_diagnostics
```

目标是解释红茶发酵阶段里 `Moderate` 和 `Over` 为什么难分，不是改主训练框架，也不是直接做最终模型。

数据来源：

```text
datas_test_point                 原始大图，按时间点 00/05/.../60 分目录
datas_test_point_30_patches      每张原图裁成 30 个 patch 后的数据
```

本诊断只使用：

```text
30, 35, 40, 45 -> moderate -> label 0
50, 55, 60     -> over     -> label 1
```

要回答三个问题：

1. 原图缩放后二分类 vs 30 patch 二分类，谁更好？判断裁剪是否破坏整体信息。
2. 同一张原图的 30 个 patch 预测是否一致？判断原图内部是否本身高度混合。
3. 颜色和纹理谁更关键？用 RGB/灰度/模糊 CNN，以及颜色直方图/LBP/GLCM + SVM 拆分信息来源。

## 已完成内容

已创建文件：

```text
.gitignore
README.md
HANDOFF.md
common.py
build_metadata.py
split_sources.py
train_binary_cnn.py
diagnose_patch_consistency.py
diagnose_color_texture.py
run_all.py
```

文件作用：

```text
common.py                         公共路径、标签、图像读取、模型、指标、绘图工具
build_metadata.py                 扫描原图和 patch，生成 metadata CSV
split_sources.py                  按 source_image_id 做 7:1:2 原图级 split
train_binary_cnn.py               训练 original/patch 的 Moderate-Over 二分类 CNN
diagnose_patch_consistency.py     聚合同一原图 30 个 patch 的预测比例
diagnose_color_texture.py         RGB/HSV/Lab/LBP/GLCM + SVM 颜色纹理诊断
run_all.py                        一键串起全部诊断流程
README.md                         简短运行说明
```

已做过的验证：

```text
所有 .py 文件语法检查通过。
build_metadata.py 可扫描数据。
split_sources.py 可生成原图级 split。
run_all.py 在 --skip-cnn --skip-features 模式下可跑通 metadata + split。
```

验证到的数据规模：

```text
原图：167 张
patch：5010 张
split：train 116 / val 17 / test 34 张原图
```

注意：还没有完整跑完 CNN 训练，正式结论必须等服务器实验结果。

## 如何运行

服务器 4090 推荐：

```bash
cd E:\workspaces\python\BlackTeaSimpleData\temp\moderate_over_diagnostics
python run_all.py --epochs 150 --patience 30 --device cuda --batch-size 64 --eval-batch-size 128
```

如果显存不够：

```bash
python run_all.py --epochs 150 --patience 30 --device cuda --batch-size 32 --eval-batch-size 64
```

快速冒烟测试：

```bash
python run_all.py --epochs 1 --patience 1 --device cpu
```

分步运行：

```bash
python build_metadata.py --run-dir outputs/debug
python split_sources.py --run-dir outputs/debug --seed 2026

python train_binary_cnn.py --run-dir outputs/debug --dataset original --variant rgb --epochs 150 --patience 30 --device cuda
python train_binary_cnn.py --run-dir outputs/debug --dataset patch --variant rgb --epochs 150 --patience 30 --device cuda
python train_binary_cnn.py --run-dir outputs/debug --dataset patch --variant gray --epochs 150 --patience 30 --device cuda
python train_binary_cnn.py --run-dir outputs/debug --dataset patch --variant blur --epochs 150 --patience 30 --device cuda

python diagnose_patch_consistency.py --run-dir outputs/debug
python diagnose_color_texture.py --run-dir outputs/debug
```

## 主要输出

默认输出在：

```text
temp/moderate_over_diagnostics/outputs/<timestamp>/
```

重点看：

```text
metadata/source_metadata.csv
metadata/patch_metadata.csv
splits/source_splits.csv
cnn_original_rgb/metrics.json
cnn_patch_rgb/metrics.json
cnn_patch_gray/metrics.json
cnn_patch_blur/metrics.json
patch_consistency/test_source_patch_consistency.csv
patch_consistency/test_top_mixed_sources.csv
color_texture/feature_summary.csv
color_texture/feature_metrics.json
```

## 结果解释

原图 vs patch：

```text
原图明显优于 patch：Moderate/Over 可能依赖整体颜色分布或空间上下文，裁剪损失信息。
原图和 patch 都低：阶段标签边界或视觉重叠本身可能是主要问题。
patch 反而更好：局部细节确实有价值。
```

30 patch 一致性：

```text
over_ratio 接近 0 或 1：同一原图内部比较一致。
over_ratio 接近 0.5：同一原图内部高度混合。
mixed_score = min(moderate_count, over_count) / patch_count，越接近 0.5 越混乱。
```

颜色/纹理：

```text
RGB 或颜色直方图明显优于灰度：颜色是主要依据。
灰度接近 RGB：纹理或形态也有较强判别力。
blur 接近 RGB：细纹理不是关键，整体颜色/大尺度信息更重要。
blur 明显下降：细纹理很重要。
所有简单特征都低：局部数据可分性有限，继续堆复杂模型可能收益不大。
```

## 当前卡点

代码功能已经完成，没有业务逻辑阻塞。

当前只剩实验未正式跑完：

```text
还没有在服务器上完整训练 CNN。
还没有根据输出结果整理论文证据图。
```

开发过程中的实际卡点是：Codex 对 E 盘写文件时审批通道多次断流。以后如果继续改本目录，建议小补丁、单文件写入、写完立刻验证。

## 下一步计划

1. 在服务器运行完整实验：

```bash
python run_all.py --epochs 150 --patience 30 --device cuda --batch-size 64 --eval-batch-size 128
```

2. 如果显存不够，降低 batch。
3. 跑完后先比较：

```text
cnn_original_rgb/metrics.json
cnn_patch_rgb/metrics.json
cnn_patch_gray/metrics.json
cnn_patch_blur/metrics.json
color_texture/feature_summary.csv
patch_consistency/test_source_patch_consistency.csv
```

4. 根据结果决定后续论文证据：
   - 原图是否明显强于 patch。
   - patch 内部是否大量 50:50 混合。
   - 颜色特征是否明显强于纹理特征。

## 绝对不要再踩的坑

1. 不要让同一张原图的 30 个 patch 跨 train/val/test。
   当前 `split_sources.py` 是按 `source_image_id` 分的，保留这个逻辑。
2. 不要把这套临时诊断并入主训练框架。
   它是问题分析工具，不是主模型 pipeline。
3. 不要只看 patch-level accuracy。
   必须看 source-level 30 patch 聚合一致性。
4. 如果未来用四分类模型做 patch 一致性，不要把预测强行限制到 Moderate/Over 二选一。
   要用原始四类预测，预测成其它类也应该算错。
5. `--device auto` 不等于一定使用 GPU。
   正式服务器实验建议写 `--device cuda`。
6. `--epochs 1` 只用于冒烟测试，不能用于结论。
   正式建议 `--epochs 150 --patience 30`。
7. E 盘写入审批通道容易断。
   后续修改文件时尽量小补丁、单文件提交。

