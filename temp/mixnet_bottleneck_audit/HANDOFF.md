# Handoff

## 目标

定位 MixNet-S 在 `00/10/20/30/40` BaSiC grid30 数据上的约 76% 瓶颈，不再添加结构模块。

## 文件

- `audit_common.py`：manifest 硬审计、数据集、MixNet-S、冻结策略、loss、patch/parent 指标。
- `run_audit.py`：B/F/L/M 与 checkpoint 上的 C 组实验。
- `analyze_features.py`：D 组 parent 内特征 spread 与类间中心距离。
- `summarize_audit.py`：跨 seed 汇总和保守 evidence matrix。
- `README.md`：运行顺序、命令和解释边界。

## 必须保留的约束

1. 数据只从 `grid30_crop_manifest.csv` 读取 parent 真值，不用正则猜 ID。
2. 每个 parent 必须恰好 30 patch、patch index 为 1–30、只属于一个 split 和一个 class。
3. 二分类仅过滤固定 split 内的类，不重新随机划分。
4. M01 是 full/CE/RGB 公共基线。
5. 冻结实验的 frozen 模块整体保持 eval（包括 BN、dropout、drop-path）；否则其状态/随机性仍会改变，不是真正的 frozen diagnostic。
6. color stress 在同一 checkpoint、同一 test patch 上评估，扰动参数由 parent+patch 的稳定 hash 决定，便于完全复现。
7. 特征脚本默认拒绝无 checkpoint；`--random-backbone` 只允许作对照/冒烟。
8. `.pth` 默认不保留；D 组需要 M01 时必须显式 `--keep-pth`。
9. 最终 train 指标使用无随机增强的独立 `train_eval_dataset`，否则随机翻转会污染 train-test 记忆差值。

## 尚未做的事

目前只交付实验设施和冒烟验证，不宣称已得到瓶颈结论。正式结论需在 GPU 上跑 3 seeds 后由 `summary.csv`、`within_vs_between.csv`、`evidence_matrix.csv` 给出。
