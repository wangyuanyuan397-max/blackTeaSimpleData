# CLAHE-L 明度增强 + 随机 55 patch 数据生成

这个文件夹用于生成一套 CLAHE 预处理后的 01234 数据集，方便和原始数据、BaSiC 数据做对比。

## 当前脚本

- `apply_clahe_then_random55.py`

## 输入与输出

默认输入：

```text
datasets_01234_original_split/
├── train/00,10,20,30,40
├── val/00,10,20,30,40
└── test/00,10,20,30,40
```

默认输出：

```text
datasets_01234_CLAHE_L_clip1p5_grid8/
├── train/00,10,20,30,40
├── val/00,10,20,30,40
├── test/00,10,20,30,40
├── _previews/clahe_l_preview.png
├── clahe_apply_manifest.csv
├── random_crop_manifest.csv
└── split_summary.json
```

## 处理原则

CLAHE 只作用在 Lab 颜色空间的 `L` 明度通道：

```text
BGR image
→ Lab
→ L channel: CLAHE
→ a/b channels: unchanged
→ Lab to BGR
→ random crop 55 patches
```

也就是说：

- 只增强局部明暗/对比度；
- 不对 RGB 三个通道分别做直方图均衡；
- 不做白平衡；
- 不改变 a/b 色度通道；
- 尽量避免把红茶本身颜色关系处理乱。

CLAHE 和 BaSiC 不同：CLAHE 是每张图独立处理，不需要从训练集拟合共享模型，所以这里不存在“只能用 train 拟合”的步骤。

## 默认参数

```python
CLAHE_CLIP_LIMIT = 1.5
CLAHE_TILE_GRID_SIZE = (8, 8)
CROPS_PER_SOURCE = 55
CROP_SIZE = 408
ENABLE_RESIZE_AFTER_CROP = False
RANDOM_SEED = 2026
```

默认不 resize，保存的是 `408×408` patch。

如果之后想生成 `224×224` 版本，把脚本顶部改成：

```python
ENABLE_RESIZE_AFTER_CROP = True
RESIZE_SIZE = 224
```

## 怎么运行

在项目根目录运行：

```powershell
python tools\data012345\CLAHE\apply_clahe_then_random55.py
```

或者在 PyCharm 里直接右键运行 `apply_clahe_then_random55.py`。

如果缺少 OpenCV：

```powershell
pip install opencv-python
```

## 第一轮建议

先只跑默认这一组：

```text
clipLimit = 1.5
tileGridSize = 8×8
```

如果默认结果不好，再做少量参数对比：

| 实验名 | clipLimit | tileGridSize | 说明 |
|---|---:|---:|---|
| CLAHE-Weak | 1.2 | 8×8 | 更保守 |
| CLAHE-Medium | 1.5 | 8×8 | 默认推荐 |
| CLAHE-Strong | 2.0 | 8×8 | 稍强，可能增强噪声 |
| CLAHE-LargeTile | 1.5 | 6×6 | 更偏大范围明暗 |
| CLAHE-SmallTile | 1.5 | 12×12 | 更偏局部纹理 |

每次改参数时，建议同步修改 `OUTPUT_ROOT`，避免不同结果混在同一个目录。

## 输出数量

如果输入是每个时间点：train 15 张、val 4 张、test 5 张原图，则输出 patch 数量为：

- train：5 个时间点 × 15 张 × 55 = 4125
- val：5 个时间点 × 4 张 × 55 = 1100
- test：5 个时间点 × 5 张 × 55 = 1375
- 总计：6600

## 绝对不要踩的坑

- 不要对 RGB/BGR 三个通道分别做 CLAHE，否则颜色关系可能被破坏。
- OpenCV 读入是 BGR，显示预览时必须转成 RGB。
- 不要把不同 CLAHE 参数的输出混到一个目录。
- 不要在输出目录非空时直接继续跑，除非你明确知道自己在覆盖/追加什么。
