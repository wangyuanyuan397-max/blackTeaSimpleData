# MixNet-S 结构搜索有效模型抽取批次

这个文件夹保存从 `temp/mixnet_structure_search/configs` 中抽出的有效结构配置，原始 `temp` 内容未删除。

运行入口：

```powershell
python tools\train_batch_01234.py
```

公共配置统一使用：

```text
configs/fixed_split_01234_BaSic_grid30_408_train.yaml
```

## 控制变量

- 数据集、01234 类别映射、BaSiC/grid30/408 输入、epoch、batch、optimizer、scheduler、loss、`keep_pth_files: false` 都来自公共配置。
- 本目录里的 YAML 只覆盖 `model`，也就是 MixNet-S 的结构搜索变量。
- 分类头仍是普通 `linear`，loss 仍是 `cross_entropy`，没有 SupCon、Margin、AD-GBC 或双视图增强。

## 去重说明

截图中有些模型名字不同，但结构内容等价，本批只保留一个：

- `stagemask_100000_k357` 等价于 `10_p10_only_s0_k357`，保留 `10_p10_only_s0_k357`。
- `stagemask_000100_k357` 等价于 `13_p13_only_s3_k357`，保留 `13_p13_only_s3_k357`。
- `stagemask_001000_k357` 等价于 `12_p12_only_s2_k357`，保留 `12_p12_only_s2_k357`。
- `stagemask_000111_k357` 等价于 `09_p09_late_s345_k357`，保留 `09_p09_late_s345_k357`。

因此截图 19 个条目最终保留 15 个唯一结构。

## 依赖代码

这些配置使用：

```yaml
backbone:
  type: mixnet_s_search
```

对应实现已经在项目代码中：

```text
src/models/backbones/mixnet_search.py
```

并且已经由：

```text
src/models/backbones/__init__.py
```

导入注册，所以运行时不再依赖 `temp/mixnet_structure_search` 下面的训练脚本或生成脚本。

## 权重文件

`tools/train_batch_01234.py` 会使用公共配置中的 `keep_pth_files: false`，并主动拒绝 `--keep-pth` / `--keep-pth-files`，避免这批复验长期保留 `.pth` 权重。
