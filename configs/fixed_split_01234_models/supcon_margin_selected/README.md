# SupCon + Margin 高分模型抽取批次

这个文件夹保存从 `temp/supcon_margin_bruteforce/configs` 中抽出的高分配置，原始 temp 内容未删除。

这些配置保留在固定目录中；如果要重新跑这一批，需要先把 `tools/train_batch_01234.py` 的 `CONFIG_LIST` 指回本目录。此前使用的运行入口是：

```powershell
python tools\train_batch_01234.py
```

该入口统一使用公共配置：

```text
configs/fixed_split_01234_BaSic_grid30_408_train.yaml
```

## 控制变量说明

- 数据集、01234 类别映射、BaSiC/grid30/408 输入、epoch、batch、scheduler、`keep_pth_files: false` 都来自公共配置。
- `supm_` 配置保留各自的模型实验变量：margin、scale、temperature、SupCon 权重、projected/raw 特征选择，以及文件名中记录的学习率。
- `baseline_mixnet_s_ce_seed2026.yaml` 不覆盖 `optimizer.lr`，直接继承公共配置中的 `lr: 0.0001`，用于和原始 MixNet-S baseline 对齐。
- SupCon 配置会覆盖 `data.train_transform` 为双视图增强；验证和测试仍使用公共配置中的单图评估增强。

## 权重文件

`tools/train_batch_01234.py` 中 `PYCHARM_KEEP_PTH_FILES = False`，公共配置里也设置了 `keep_pth_files: false`，跑完不会长期保留 `.pth` 权重文件。

另外，`tools/train_batch_01234.py` 会主动拒绝 `--keep-pth` / `--keep-pth-files`，避免命令行误传后覆盖这批复验规则。
