# Global-Local Multi-Scale 双分支消融

本组实验在 EfficientNetV2-S 的最终特征图 `X` 上同时建模两类信息：

- global branch：`GAP(X) -> g`，描述当前输入 patch 的整体颜色和发酵状态；
- local branch：`1x1 Conv -> 3/5/7 DWConv -> Concat -> 1x1 Conv -> GAP -> l`，描述局部纹理、暗化和堆积差异。

这里的“global”只表示当前输入 patch 内的全局信息，不代表裁剪前原图的全局信息。

## 实验列表

- `glms_baseline_gap.yaml`：只使用 global GAP，是本组结构对照。
- `glms_local_only.yaml`：只使用多尺度局部分支。
- `glms_concat_fusion.yaml`：直接拼接 `g` 与 `l`。
- `glms_gated_fusion.yaml`：按样本动态学习 global/local 两个分支的权重。
- `glms_concat_adaptive_soft.yaml`：保持 concat 结构不变，只把 hard CE 改为边界距离自适应 soft label。

前三个结构实验和 gated 实验都使用相同的 hard-label CE。建议先比较前四组；只有当 concat 确实有稳定信号时，再把最后一组用于判断结构与有序监督是否互补。

公共训练参数、数据划分、随机种子、输入变换、优化器和早停设置均继承 `configs/fixed_split_patches_train.yaml`。

当前公共输入尺寸是 224，EfficientNetV2-S 最终特征图通常较小。论文中不要把最终空间尺寸写死，应以实际运行时打印结果为准。
