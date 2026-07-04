# Controlled Single-Variable Experiments

这组配置用于四分类红茶发酵程度的严格单变量实验。所有首轮实验继承
`configs/fixed_split_patches_train.yaml` 中相同的数据划分、增强、优化器、训练轮数、
选模指标和随机种子 2026。

实验顺序：

1. `full_effv2s_ce` 与 `stage5_ce`：只比较最终特征和 Stage5 特征。
2. `stage5_coral`、`stage5_corn`：只改变有序边界建模方式。
3. `stage5_ce_adjcon_m0.5_lam0.05`：只增加相邻类别对比损失。
4. `stage5_ce_finetune_uniform` 与 `stage5_ce_finetune_hardadj_w2`：都从同一个
   `stage5_ce` 最佳 checkpoint 开始，使用相同的 20 epoch、学习率和优化器；唯一差异是
   困难相邻样本权重为 1 或 2。即使关闭 PTH 保留，批量脚本也会在内存中传递来源权重。
5. `stage5_beta_nll_reg1e-4`：只把 CE 分类头替换为 Beta-NLL 概率有序头。

`Acc_0_1`、`Acc_1_2`、`Acc_2_3` 只按真实标签筛选对应两类，预测仍使用原始四分类
结果；预测到该类别对之外也计错，不进行二选一重归一化。

第一轮不组合 Stage5、Beta-NLL、相邻对比和 hard-adjacent weighting。只有单变量信号
经过多随机种子复现后，才考虑组合实验。
