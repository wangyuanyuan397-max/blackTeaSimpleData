# fixed_split_01234_models

这个目录是为 `datasets_01234` 生成的五分类模型配置副本。

- 来源 1：`configs/fixed_split_patches_models`，共 13 个模型。
- 来源 2：`configs/tryPractice/Attention`，共 31 个 attention 位置消融模型。
- 所有硬编码的 `num_classes: 4` 已改成 `num_classes: 5`。
- 文件名前加了 `fixed_` 或 `attention_` 前缀，避免两个来源出现同名 YAML 时冲突。
- 配套公共训练配置：`configs/fixed_split_01234_train.yaml`。
- 推荐运行入口：`tools/train_batch_01234.py`。

总实验数：44。
