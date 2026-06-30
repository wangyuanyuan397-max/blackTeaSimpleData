# Ordinal class-rank 与 OPCL 实验

本目录只使用四个类别的自然顺序：

- Pre = 0，rank target = 0
- Slight = 1，rank target = 1/3
- Moderate = 2，rank target = 2/3
- Over = 3，rank target = 1

这里没有 13 个时间点，因此 rank 分支是 ordinal stage-rank regression，
不是 temporal/time-rank regression。

## 实验批次

- 第一批 V0～V4：baseline、projection control、hard prototype、ordinal soft prototype、OPCL。
- 第二批 V5～V8：class-rank 及其与 prototype/OPCL 的组合。
- 第三批 V9～V11：classification soft label、soft label + rank、完整 soft label + rank + OPCL。

所有配置已经追加在 tools/train_batch.py 的 CONFIG_LIST 末尾，前面的实验顺序保持不变。
如果不想一次运行全部实验，可以使用 --models 指定本目录中的实验名。

## 默认参数

- lambda_rank = 1.0
- alpha_proto = 0.1
- gamma = 0.1
- temperature = 0.1
- embedding_dim = 128

推理始终只使用 classification logits。rank 和 prototype 分支只参与训练约束。

V0、V2、V4、V8、V11 会额外保存 representation_features.pt、t-SNE 文件；
含 prototype 的版本还会保存 prototype cosine distance 矩阵。
