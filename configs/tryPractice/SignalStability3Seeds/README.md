# 五个候选信号的三种子稳定性复现

用户给出的清单包含 5 个模型设置，每个使用 seed 1、2、3，因此实际为
15 次独立训练：

- baseline_ce
- stage_probe_s5
- beta_nll_center_reg1e-4
- beta_cdf_grid_ce_offset0.5
- logistic_normal_cdf_ce

除 random_seed 外，数据划分、增强、优化器、学习率、epoch、early stopping
和模型选择指标均读取同一个公共训练配置。

批量完成后会输出：

- signal_stability_3seeds_summary.csv：15 次逐实验结果。
- signal_stability_3seeds_folds.csv：明确标记 fixed_split。
- seed_reproduction_summary.csv：按五个模型族计算 Accuracy、Macro-F1、
  MAE、QWK 的 mean 和 sample standard deviation。
