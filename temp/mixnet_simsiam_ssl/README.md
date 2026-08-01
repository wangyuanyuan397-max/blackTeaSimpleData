# MixNet-S SimSiam 自监督实验

这个临时实验面向：

- 数据集：`datasets_01234_BaSic`
- 类别：`00`、`10`、`20`、`30`、`40`
- backbone：`timm` 模型 `mixnet_s`
- 第 1 阶段：不使用标签的 SimSiam 自监督预训练
- 第 2 阶段：加载自监督 backbone 后进行有监督五分类微调

所有新增代码都放在当前文件夹中，运行产物会写到
`temp/mixnet_simsiam_ssl/runs/`。

## 1. 无标签预训练

默认命令。它只把 `datasets_01234_BaSic/train` 作为无标签图片使用，
因此验证集和测试集图片不会在正式评估前被模型看到。

```powershell
python temp\mixnet_simsiam_ssl\train_simsiam_pretrain.py
```

主要输出是：

```text
temp/mixnet_simsiam_ssl/runs/simsiam_mixnet_s_basic408/mixnet_s_simsiam_backbone.pth
```

如果显存紧张，可以减小 batch size：

```powershell
python temp\mixnet_simsiam_ssl\train_simsiam_pretrain.py --batch-size 8
```

如果你明确希望把所有划分都作为无标签图片用于 SSL：

```powershell
python temp\mixnet_simsiam_ssl\train_simsiam_pretrain.py --ssl-splits train val test
```

这样可能提升表征学习效果，但它属于 transductive 设置，
因为测试图片虽然没有标签，但已经在自监督阶段被模型看过。

## 2. 有标签微调

等 SSL checkpoint 存在后再运行：

```powershell
python temp\mixnet_simsiam_ssl\finetune_mixnet_classifier.py
```

重要输出：

```text
temp/mixnet_simsiam_ssl/runs/finetune_mixnet_s_basic408/best_model.pth
temp/mixnet_simsiam_ssl/runs/finetune_mixnet_s_basic408/summary.json
temp/mixnet_simsiam_ssl/runs/finetune_mixnet_s_basic408/history.csv
```

`summary.json` 包含最佳验证集指标和最终测试集指标。

## 3. 微调调参搜索

你当前的 SSL 结果更像欠拟合，而不是过拟合：训练集和验证集准确率都在
0.60 左右。因此第一轮调参固定 SSL checkpoint，只放宽有监督微调阶段：

- 更弱的训练增强：`resize_flip` 或更温和的 crop
- 更低的 weight decay
- 更高的 AdamW 学习率
- 可选：给分类头使用更快的学习率
- 可选：针对有序任务使用 `val_qwk` 选择 checkpoint

列出推荐变体：

```powershell
python temp\mixnet_simsiam_ssl\run_finetune_sweep.py --list
```

只预览命令，不实际运行：

```powershell
python temp\mixnet_simsiam_ssl\run_finetune_sweep.py --dry-run --exclude-sgd
```

优先运行推荐的 AdamW 变体：

```powershell
python temp\mixnet_simsiam_ssl\run_finetune_sweep.py --exclude-sgd
```

运行某一个指定变体：

```powershell
python temp\mixnet_simsiam_ssl\run_finetune_sweep.py --variant resize_lr3e4_wd1e4_acc
```

搜索结果会写到：

```text
temp/mixnet_simsiam_ssl/runs/tuning/sweep_results.csv
```

每个变体也会在 `temp/mixnet_simsiam_ssl/runs/tuning/{variant_name}/`
下面生成自己的 `summary.json`、`history.csv` 和 `best_model.pth`。

## 快速冒烟检查

下面这些命令只检查流程能否正确实例化模型和 batch：

```powershell
python temp\mixnet_simsiam_ssl\train_simsiam_pretrain.py --dry-run --image-size 224 --batch-size 4 --num-workers 0 --max-samples 8
```

如果没有真实 SSL checkpoint，也可以这样做微调冒烟测试：

```powershell
python temp\mixnet_simsiam_ssl\finetune_mixnet_classifier.py --dry-run --ssl-checkpoint none --image-size 224 --batch-size 4 --eval-batch-size 8 --num-workers 0 --max-samples 20
```

## 备注

- 默认 `image_size=408`，与当前 BaSiC/grid30 MixNet baseline 保持一致。
- SSL 脚本不依赖 `lightly`；SimSiam loss 和 head 都是直接用 PyTorch 实现的。
- Windows 多进程要求 dataloader worker helper 可以被 pickle。脚本支持
  `--num-workers 4`；如果本机 worker 启动仍有问题，可以传
  `--num-workers 0` 改成单进程加载。
- 默认 SSL 从随机初始化的 MixNet-S 权重开始。如果希望 SSL 前先加载
  ImageNet 初始化，可以加 `--imagenet-pretrained`。
- 微调默认使用 AdamW，`lr=1e-4`，与现有 BaSiC baseline 配置对齐。
  如果想尝试粘贴方案里的 SGD 风格，可以使用
  `--optimizer sgd --lr 0.001`。
