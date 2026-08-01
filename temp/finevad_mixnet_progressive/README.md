# FineVAD 风格 MixNet-S 渐进式三阶段实验

这是给 `datasets_01234_BaSic` 单独开的实验沙盒。新增代码都放在 `temp/finevad_mixnet_progressive`，没有改动主线 `src/` 训练框架。

## 实验内容

- 骨干网络：`timm` 里的 `mixnet_s`
- 细粒度 5 类：`00, 10, 20, 30, 40`
- 粗粒度 2 类：`{00,10}->0`，`{20,30,40}->1`
- 中粒度 3 类：`{00,10}->0`，`{20,30}->1`，`{40}->2`
- 最终验证和测试：只看 fine head 的 5 分类输出

## 三阶段

1. `stage1_coarse`：训练 backbone + coarse head，损失为粗粒度 CE + 监督对比学习。
2. `stage2_mid_fusion`：训练 coarse-to-mid 融合模块和 mid head，不使用硬中粒度伪标签 CE，只用 coarse 概率的一致性软引导。
3. `stage3_fine`：全模型细粒度训练，使用 fine CE，并额外惩罚同一中粒度组内的混淆，比如 `00<->10`、`20<->30`。

## 命令

先检查数据路径、类别映射、模型输出形状；关闭预训练，避免联网下载：

```powershell
conda run -n yolov8 python temp\finevad_mixnet_progressive\train_finevad_mixnet.py --dry-run --no-pretrained --batch-size 2 --eval-batch-size 2 --num-workers 0
```

跑一个极小 smoke training：

```powershell
conda run -n yolov8 python temp\finevad_mixnet_progressive\train_finevad_mixnet.py --no-pretrained --epochs-stage1 1 --epochs-stage2 1 --epochs-stage3 1 --batch-size 2 --eval-batch-size 2 --num-workers 0 --max-train-batches 2 --max-eval-batches 2
```

跑正式实验：

```powershell
conda run -n yolov8 python temp\finevad_mixnet_progressive\train_finevad_mixnet.py --device cuda
```

输出目录：`temp/finevad_mixnet_progressive/runs_BaSic/<run_name>/`。
