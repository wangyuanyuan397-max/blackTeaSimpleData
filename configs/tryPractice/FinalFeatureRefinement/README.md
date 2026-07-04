# Final Feature Refinement Experiments

这组实验复用已经完成的 EfficientNetV2-S baseline，只新增 M3～M6 四个模型。四组实验均继承
`configs/fixed_split_patches_train.yaml` 的数据划分、增强、优化器、150 epoch、选模指标和
随机种子 2026。

统一主干结构：

```text
Input
→ EfficientNetV2-S final feature map X（1280 × 7 × 7，输入 224 时）
→ refinement module
→ GAP
→ Dropout(0.2)
→ Linear(4)
```

所有 refinement 都严格位于 GAP 前面：

- `m3_effv2s_msr`：`1×1 reduce → DWConv 3/5/7 branches → concat → 1×1 fuse
  → 1×1 expand → X + γF(X)`。
- `m4_fmr_efficientnet_msr_eca`：与 M3 使用完全相同的 MSR，再接 ECA。论文候选名为
  **FMR-EfficientNet（Fermentation-aware Multi-scale Refinement EfficientNet）**。
- `m5_effv2s_dcn`：`1×1 reduce → offset conv → torchvision DeformConv2d 3×3
  → BN + SiLU → 1×1 expand → X + γF(X)`。offset 卷积零初始化，开始时采用规则采样。
- `m6_effv2s_msr_se`：与 M4 使用完全相同的 MSR，只把 ECA 替换为 SE。

统一设置：`refine_channels=256`、可学习残差系数 `γ` 初值为 `0.1`、hard-label CE。
M4 与 M6 的唯一结构差异是 ECA/SE，便于直接判断注意力形式；M3 与 M4 的差异只有 ECA。
