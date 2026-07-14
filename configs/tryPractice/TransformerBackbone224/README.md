# TransformerBackbone224

这组配置用于比较 Transformer / hybrid backbone 在同一套 224x224 patch 数据、同一套 hard-label CE 监督下的迁移效果。

统一设置：

- 输入为 224x224 RGB patch。
- backbone 使用 `timm.create_model(..., pretrained=true, num_classes=0)` 提取特征。
- 外接项目统一的 `linear` head 输出 4 类 logits。
- loss 继承公共配置 `configs/fixed_split_patches_train.yaml` 中的 `cross_entropy`。
- CrossEntropyLoss 前不做 Softmax。

当前接入 3 个可以直接通过 timm 创建的模型：

- `fastvit_sa24_256pre_224ft_ce`: `fastvit_sa24.apple_in1k`
- `efficientformerv2_s2_ce`: `efficientformerv2_s2.snap_dist_in1k`
- `shvit_s3_ce`: `shvit_s3.in1k`

注意：FastViT-SA24 的 ImageNet-1K 官方预训练输入为 256x256，本实验仍使用统一的 224x224 下游输入，因此命名中写为 `256pre_224ft`。EfficientFormerV2-S2 的 checkpoint 名中带 `dist`，表示 ImageNet 预训练阶段使用过蒸馏；红茶下游训练仍然只是普通 CE。

RepViT-M2.3 已经在 `configs/tryPractice/TimmBackbone224` 中配置过，不在这一组重复计数。
