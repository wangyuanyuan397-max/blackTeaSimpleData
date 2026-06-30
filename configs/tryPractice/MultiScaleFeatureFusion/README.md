# Multi-scale 与 Feature Fusion Neck 消融配置

本目录的所有模型共享 configs/fixed_split_patches_train.yaml 中的数据、训练和评估设置。

## Stage 定义

- Stage 1：EfficientNetV2-S stem + 第一组 block，输出 24 通道。
- Stage 2：输出 48 通道。
- Stage 3：输出 64 通道。
- Stage 4：输出 128 通道。
- Stage 5：输出 160 通道。
- Stage 6：最后一组 block + 1×1 head conv，输出 1280 通道，然后进入 GAP。

模型不会修改 MBConv 或 Fused-MBConv 内部结构。

## 目录

- baseline.yaml：官方 backbone 特征直接执行 GAP + Linear。
- multiscale：MSK、MSD、MSP、MSH 在 P3/P4/P5/P6/P46/P56/P456 的 28 组 concat 实验。
- neck：F36/F46/F56/F346/F456/F3456 与 concat/add/weighted/attention 的 24 组实验。
- combined：方案中优先推荐的 MSD-P6 + concat-F46 候选组合。

bestP 和 bestF 必须根据前一轮实验结果决定。模型代码已经支持
multiscale_fusion 的 concat/add/gated，以及
neck_fusion 的 concat/add/weighted/attention。确定最佳位置后复制对应 YAML 修改字段即可。

当前任务使用固定 train/val/test 划分和交叉熵，不启用 LOGOCV、soft-label 或 regression head。
folds.csv 中的 fold 会明确写成 fixed_split。
