# Global-Local Attention 第一阶段消融

这一组实验只验证一个问题：Moderate / Over 的判别信息是否集中在 EfficientNetV2-S 最后 feature map 的局部空间区域。

为了保证控制变量干净，本组实验保持：

- same dataset split
- same input transform
- same optimizer / scheduler / epochs
- same hard-label CE
- same EfficientNetV2-S ImageNet pretrained backbone

只改变 pooling 方式。

## 实验列表

- `gla_baseline_gap.yaml`：普通 GAP，对照组。
- `gla_local_only.yaml`：只使用 spatial-softmax local pooling 得到的局部特征。
- `gla_global_local.yaml`：使用 `global_feature + gamma * local_feature`，其中 `gamma` 初始为 0。

## 注意

当前项目公共 transform 会把 patch resize 到 224，因此最后 feature map 的空间尺寸由实际输入尺寸决定。不要在论文中未经打印确认直接写死为 16×16。
