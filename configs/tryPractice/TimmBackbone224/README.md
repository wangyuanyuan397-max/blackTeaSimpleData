# TimmBackbone224

这组配置用于比较不同 ImageNet-1K 预训练 backbone 在同一套 224x224 patch 数据、同一套 hard-label CE 监督下的迁移效果。

所有模型都保持：

- 输入为 224x224 RGB patch。
- backbone 使用 `timm.create_model(..., pretrained=true, num_classes=0)` 提取特征。
- 外接项目统一的 `linear` head 输出 4 类 logits。
- loss 继承公共配置 `configs/fixed_split_patches_train.yaml` 中的 `cross_entropy`。
- CrossEntropyLoss 前不做 Softmax。

当前先接入最容易直接运行的 5 个 timm 模型：

- `convnextv2_tiny_ce`: `convnextv2_tiny.fcmae_ft_in1k`
- `fasternet_t2_ce`: `fasternet_t2.in1k`
- `inceptionnext_tiny_ce`: `inception_next_tiny.sail_in1k`
- `repvit_m2_3_ce`: `repvit_m2_3.dist_450e_in1k`
- `mambaout_tiny_timm_ce`: `mambaout_tiny.in1k`

InternImage、FlashInternImage、OverLoCK、ShiftWiseConv 涉及官方仓库或自定义 CUDA 算子，建议后续单独环境处理，不混进当前主训练环境。
