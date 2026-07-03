# Probing、容量减法、Stage 截断与多种子复现

本目录包含 26 个实验，并追加在 tools/train_batch.py 的 CONFIG_LIST 最后。

## probing

- probe_freeze_all：冻结 Stage1～6，只训练 Linear 分类器。
- probe_unfreeze_s6：只训练 Stage6 和分类器。
- probe_unfreeze_s56：只训练 Stage5～6 和分类器。
- probe_unfreeze_s456：只训练 Stage4～6 和分类器。

被冻结 stage 的参数和 BatchNorm 运行统计都会固定。

## capacity

比较 tf_efficientnetv2_b0/b1、efficientnet_v2_s、efficientnet_b0/b1、
MobileNetV3-Large 和 ResNet18。全部使用 ImageNet 预训练和统一线性分类头。

## stage_probe

- stage_probe_s4：只保留 Stage1～4，使用 X4。
- stage_probe_s5：只保留 Stage1～5，使用 X5。
- stage_probe_s6：完整 Stage1～6，使用 X6。

被截断的更深 stage 不参与参数量和 FLOPs 统计。

## seed_reproduction

baseline、CA-P46、neck-concat-F56、CE-OPCL 各使用 seed 1、2、3 重复运行。
每个 YAML 顶层的 random_seed 会覆盖公共随机种子。

## PTH 保存开关

configs/fixed_split_patches_train.yaml 中：

- keep_pth_files: false：最佳权重完成测试和报告后删除，默认用于节省空间。
- keep_pth_files: true：保留 best_model.pth。

该开关不会跳过最佳 checkpoint 测试，只控制所有评估完成后是否保留文件。
