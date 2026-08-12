# 红茶五分类任务诊断实验

这是一套与主工程隔离的诊断性尝试。代码全部位于本目录，不注册 `src` 组件，不修改现有训练配置。目标不是继续给 MixNet-S 堆模块，而是先用受控实验回答：判别证据依赖什么、尺度是否互补、空间布局与颜色/纹理是否重要。

## 已实现的证据链

1. **Global 与物理视野尺度**：整张原图 resize 作为 Global；另从同一原图随机裁出 `408/306/204/102` 像素视野，全部 resize 到 `224×224`，分别训练同一个 MixNet-S。
2. **尺度互补**：独立模型的测试概率做配对平均，自动比较两尺度与全尺度融合。
3. **单局部与多局部**：每个局部尺度同时报告中心单 crop 和固定多 crop 概率平均，避免把视野尺度与集成增益混为一谈。
4. **局部证据分布**：对同一 checkpoint 做 `10%～50%` 的随机矩形遮挡，默认重复 3 次。
5. **空间关系**：`3×3 patch shuffle` 保留局部像素内容但破坏全局布局。
6. **颜色与纹理**：灰度、Gaussian blur、确定性 color jitter，以及“粗区域内细块打乱”的 texture shuffle。
7. **有序错误**：每个条件自动输出 confusion matrix、MAE、QWK、相邻错误占比和远距离错误占比。

所有随机破坏都由“图片路径 + 条件 + 重复编号 + seed”生成稳定随机数，重复执行不会悄悄换一套扰动。
所有扰动都在 resize 后的模型输入画布上执行，因此 Global 与 Local 的 blur、shuffle、遮挡强度具有同一像素尺度。

## 为什么默认用原图目录

默认数据为：

```text
datasets_01234_original_split/
├── train/00..40
├── val/00..40
└── test/00..40
```

该目录每类为固定划分的原始 `2448×2048` 图像。同一原图随机产生多个训练 crop，但 train/val/test 间不会共享原图。这样比较不同 crop 尺度时，改变的是“模型看到的物理范围”，不是提前生成的数据集差异。

如果改用 `datasets_01234_408` 等派生 patch 数据，工具仍会按文件名恢复原图 ID，并同时输出：

- `sample_*`：patch 级指标；
- `parent_*`：同一原图所有 patch 概率平均后的原图级指标。

正式结论优先使用 `parent_*`。否则 55 个高度相关 patch 会被误当成 55 个独立样本。

## 0. 环境与帮助

在项目根目录执行：

```powershell
python temp/problem_diagnosis_experiments/run_crop_experiments.py --help
python temp/problem_diagnosis_experiments/run_perturbation_experiments.py --help
python temp/problem_diagnosis_experiments/summarize_diagnosis.py --help
python temp/problem_diagnosis_experiments/test_diagnosis_common.py
```

依赖复用项目现有 `requirements.txt`，无需安装额外包。

## 1. 先做冒烟测试

冒烟测试只读取每类 1 张图，并只跑前向，不下载预训练权重：

```powershell
python temp/problem_diagnosis_experiments/run_crop_experiments.py `
  --crop-sizes 408,204 `
  --train-repeats 1 `
  --val-views 1 `
  --test-views 1 `
  --batch-size 2 `
  --eval-batch-size 2 `
  --num-workers 0 `
  --max-samples-per-class 1 `
  --no-pretrained `
  --dry-run
```

默认还会检查 `global` 条件；如只想检查局部尺度，可加 `--skip-global`。

## 2. 正式训练尺度诊断模型

```powershell
python temp/problem_diagnosis_experiments/run_crop_experiments.py `
  --crop-sizes 408,306,204,102 `
  --input-size 224 `
  --epochs 150 `
  --train-repeats 30 `
  --val-views 5 `
  --test-views 9 `
  --batch-size 32 `
  --eval-batch-size 16 `
  --device auto
```

说明：

- `train-repeats=30` 表示每轮每张原图产生 30 个随机 crop，不会在磁盘复制数据；
- `crop_global` 保留整张原图后 resize，作为真正的全局对照；
- 每个局部 checkpoint 会分别生成 `test_single` 与 `test_multi<N>` 结果；
- 各尺度都送入相同的 `224×224` 网络输入，架构与像素张量大小保持一致；
- 各尺度重新设置相同随机种子；
- early stopping 与最佳权重选择使用验证集原图聚合准确率；
- 最优权重保存为 `results/crop_scale/crop_<size>/best.pth`；
- `summary.csv` 同时含单局部、多局部与概率融合结果。
- 各指标附带 Wilson 95% 区间；扰动结果另含同一批原图的配对得失数和精确 McNemar p 值。

原图很大、训练 IO 压力高。如显存或磁盘吞吐不足，可先把 `train-repeats` 调到 10、`batch-size` 调小；正式对比时所有尺度必须用相同值。

## 3. 受控破坏实验

以某个尺度的最佳 checkpoint 为例：

```powershell
python temp/problem_diagnosis_experiments/run_perturbation_experiments.py `
  --checkpoint temp/problem_diagnosis_experiments/results/crop_scale/crop_204/best.pth `
  --split test `
  --views 9 `
  --repeats 3 `
  --device auto
```

默认运行：

```text
Original
Occlusion 10%, 20%, 30%, 40%, 50%
Grayscale
Gaussian blur radius 1.5, 3.0
Color jitter strength 0.3
Patch shuffle 3×3
Texture shuffle macro 2×2, micro 4×4
```

随机条件按重复编号分别保存，`summary.csv` 给出相对 Original 的原图准确率下降。

建议先在 `--split val` 上确定扰动强度；测试集只做最后一次冻结评估，避免反复窥视测试结果。

## 4. 生成诊断报告

```powershell
python temp/problem_diagnosis_experiments/summarize_diagnosis.py
```

生成：

```text
results/DIAGNOSIS_REPORT.md
results/DIAGNOSIS_REPORT.json
```

报告会：

- 找出最佳单尺度和最佳尺度融合；
- 计算融合相对最佳单尺度的增益；
- 对随机扰动重复计算均值和标准差；
- 明确提醒哪些现象只能算“初步信号”，不能直接写成机制结论。

## 5. 结果怎么解释

建议使用以下证据组合，而不是孤立看一个数字：

| 假设 | 至少需要的互补证据 |
| --- | --- |
| 局部证据重要 | 中等 crop 优于大视野；遮挡稳定掉点；多视图优于单视图 |
| 全局与局部互补 | 单尺度各自有效；融合稳定优于最佳单尺度；不同 seed 可复现 |
| 空间布局重要 | patch shuffle 在重复后稳定大幅下降；但颜色直方图等仍基本保留 |
| 纹理重要 | blur/texture shuffle 稳定下降；灰度结果可帮助区分颜色与纹理贡献 |
| 相邻阶段边界模糊 | 相邻错误占全部错误比例高；MAE/QWK 与 confusion matrix 一致支持 |

不要把 `Local branch +2%`、某次遮挡掉点或一次融合涨点单独写成因果证明。正式结论建议补 3 个训练随机种子，并以原图为独立单位做配对统计。

## 当前范围

本次先完成附件中成本低、能直接形成证据链的 crop/扰动/混淆诊断。`random split vs batch-held-out` 需要可靠的批次字段；当前固定数据目录与 manifest 只有时间点和原图编号，没有明确独立批次标签，因此没有根据文件名猜测批次，避免生成伪结论。
