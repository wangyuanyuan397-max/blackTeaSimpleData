# BaSiC 明度照明矫正数据处理

本文件夹用于生成 `datasets_01234_BaSic` 数据集。

核心流程：

1. 输入 `datasets_01234_original_split` 中的完整原图。
2. 只使用 `train` split 的完整原图拟合 BaSiC 照明场。
3. `val` 和 `test` 不参与 BaSiC fit，避免预处理阶段的数据泄露。
4. 将 train 拟合出的同一个照明场应用到 `train/val/test`。
5. 对每张矫正后的完整原图随机裁剪 55 个 `408x408` patch。
6. 默认不 resize，直接输出 `408x408` patch。

## 只矫正明度

本方案只做单通道明度照明矫正：

```text
gain(x, y) = median(flatfield) / flatfield(x, y)
R' = gain(x, y) * R
G' = gain(x, y) * G
B' = gain(x, y) * B
```

也就是说：

- 不分别拟合 R/G/B 三个通道；
- 不做白平衡；
- 不做颜色直方图均衡；
- 不改变像素内部的 R:G:B 比例；
- 只修正空间位置上的亮暗不均。

## 运行方式

```powershell
cd E:\workspaces\python\BlackTeaSimpleData
python tools\data012345\BaSiC\apply_basic_then_random55.py
```

如果缺少依赖：

```powershell
pip install basicpy opencv-python
```

## 默认输入输出

输入：

```text
datasets_01234_original_split/
  train/00,10,20,30,40
  val/00,10,20,30,40
  test/00,10,20,30,40
```

输出：

```text
datasets_01234_BaSic/
  train/00,10,20,30,40
  val/00,10,20,30,40
  test/00,10,20,30,40
```

每类 patch 数量应为：

```text
train: 15 * 55 = 825
val:   4  * 55 = 220
test:  5  * 55 = 275
```

## 输出记录文件

脚本会在 `datasets_01234_BaSic` 下保存：

- `basic_fit_manifest.csv`：参与 BaSiC fit 的 train 原图清单；
- `basic_apply_manifest.csv`：被应用照明矫正的 train/val/test 原图清单；
- `random_crop_manifest.csv`：每个 patch 的父图和裁剪坐标；
- `split_summary.json`：处理参数和数量汇总；
- `flatfield_small.npy`：训练集拟合得到的单通道照明场；
- `flatfield_preview.png`：照明场和 gain 预览；
- `basic_correction_preview.png`：矫正前后对照图。

## 重要约束

不要把 `val` 或 `test` 原图加入 BaSiC fit。虽然 BaSiC 不使用标签，但测试图像参与预处理参数估计仍然可能被认为是数据泄露。
