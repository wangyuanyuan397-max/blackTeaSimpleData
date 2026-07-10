# Boundary-proximity adaptive soft label 实验组

这个目录只比较标签监督方式，不改模型结构。所有 YAML 都使用相同的 EfficientNetV2-S ImageNet 预训练 backbone，区别只在 `loss`。

## 实验列表

- `boundary_baseline_ce.yaml`：hard label CE 基线。
- `boundary_fixed_adjacent_soft_eps007.yaml`：固定相邻 soft label，对照你之前的固定 ε=0.07 思路。
- `boundary_adaptive_soft_eps010_tau05.yaml`：自适应 soft label，`eps_max=0.10`。
- `boundary_adaptive_soft_eps015_tau05.yaml`：自适应 soft label，`eps_max=0.15`。
- `boundary_adaptive_soft_eps020_tau05.yaml`：自适应 soft label，`eps_max=0.20`，推荐主参数。
- `boundary_adaptive_soft_eps025_tau05.yaml`：自适应 soft label，`eps_max=0.25`。

## 自适应标签规则

文件名前缀会被解析成发酵时间：

```text
00 -> 0.0 h
05 -> 0.5 h
10 -> 1.0 h
...
60 -> 6.0 h
```

阶段边界固定为：

```text
Pre / Slight:       1.25 h
Slight / Moderate: 2.75 h
Moderate / Over:   4.75 h
```

越靠近边界，相邻类别概率越高；越远离边界，标签越接近 one-hot。
