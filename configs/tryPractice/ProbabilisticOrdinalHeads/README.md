# Probabilistic Ordinal Head 实验包

本目录包含 1 个标准 baseline 和 12 个概率有序头实验，统一使用
EfficientNetV2-S、固定四个区间边界以及公共训练配置。

阶段区间：

- Pre：[0, 0.25)
- Slight：[0.25, 0.50)
- Moderate：[0.50, 0.75)
- Over：[0.75, 1]

## Beta

- beta_cdf_grid_ce：offset=1.0，200 点 midpoint grid。
- beta_cdf_grid_ce_offset0.5：允许更灵活的边界形状。
- ce_beta_cdf_aux_lam0.1/0.3/0.5：普通 CE 主头 + Beta-CDF 辅助损失。
- beta_nll_center_reg1e-4/1e-3：使用四类人为中心点的 Beta NLL。
- ce_beta_nll_aux_lam0.1：普通 CE 主头 + Beta center NLL。

## 分布对照

- kuma_cdf_ce / ce_kuma_cdf_aux_lam0.3
- logistic_normal_cdf_ce / ce_logistic_normal_aux_lam0.3

纯概率模型使用分布区间概率推理；纯 Beta-NLL 使用分布均值按
0.25/0.50/0.75 分级。所有 ce_* 辅助模型始终使用 Linear(4) 推理。

每个概率实验会保存 test_predictions_probabilistic.csv，其中包含路径、
真实/预测标签、四类概率、分布参数、max probability 和 Beta variance。
