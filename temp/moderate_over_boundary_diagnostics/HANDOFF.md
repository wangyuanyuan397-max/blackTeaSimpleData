# 会话交接：Moderate ↔ Over 多模型边界诊断

更新时间：2026-07-13（Asia/Shanghai）

## 新会话先读这里

当前代码已经完成多模型诊断改造并推送到远端，但**本工作区没有任何正式实验结果可供分析**。`temp/moderate_over_boundary_diagnostics/results/` 当前不存在，而且该目录被 `.gitignore` 忽略，不会随 Git 推送同步。

因此，接手后不要声称 17 组正式训练已经完成。第一件事应确认用户所说“刚刚跑的实验”究竟运行在哪台机器、哪个目录；如果无法找到结果，再按本文的“小范围验证 → 单模型正式运行 → 扩大运行”顺序继续。

## 我们在做什么

目标是诊断红茶四级分类中 `Moderate(2) ↔ Over(3)` 的相邻边界混淆。实验固定使用项目已有的 `datasets_split_patches/{train,val,test}`，不重新拆分数据，不修改正式训练入口、正式模型 YAML 或原有 `runs/`。

需要比较以下四个模型：

1. `configs/fixed_split_patches_models/mambaout_tiny.yaml`
2. `configs/fixed_split_patches_models/resnet50.yaml`
3. `configs/fixed_split_patches_models/convnext_tiny.yaml`
4. `configs/fixed_split_patches_models/safnet_imagenet.yaml`

临时入口是：

```text
temp/moderate_over_boundary_diagnostics/run_experiments.py
```

相关说明是：

```text
temp/moderate_over_boundary_diagnostics/README.md
```

## 已经完成了什么

### 1. 四个模型已接入列表

`run_experiments.py` 中的 `MODEL_CONFIG_LIST` 已包含上述四个 YAML。脚本会从 YAML 构建真实模型，不只是借用模型名称。

模型结构上的关键区别已经处理：

- ResNet50、ConvNeXt-Tiny 的分类器是项目外部 `LinearHead`，类别数需要写入 `head.num_classes`。
- MambaOut、SAFNet 的分类器包含在 backbone 内，二分类必须同时覆盖 `backbone.num_classes=2`。
- 构建前会 `deepcopy` YAML 字典，因为项目的 `ImageClassifier` 构造过程会从嵌套配置中 `pop("type")`，直接复用原字典会被修改。

### 2. 默认作业矩阵已确定

每个 YAML 模型依次运行四个兼容实验：

1. `four_class_ce`
2. `binary_moderate_over_ce`
3. `four_class_ce_mo_weight1.5`
4. `four_class_ce_mo_logit_margin_m0.3_lam0.1`

之后独立运行：

5. `logistic_normal_cdf_t23_search`

默认总数为：

```text
4 个模型 × 4 个兼容实验 + 1 个专用 Logistic-Normal 实验 = 17 个作业
```

作业串行执行。默认右键运行会启动全部 17 个作业，不是只跑当前光标所在的实验。

### 3. 各实验的含义

#### `four_class_ce`

普通四分类交叉熵，类别为：

```text
pre=0, slight=1, moderate=2, over=3
```

#### `binary_moderate_over_ce`

每个固定 split 只保留真实标签 2/3，并映射为：

```text
moderate=0, over=1
```

这是独立二分类任务，其 Accuracy 不应与四分类 Accuracy 直接横向比较。

#### `four_class_ce_mo_weight1.5`

仍是四分类，但交叉熵类别权重为：

```text
[1.0, 1.0, 1.5, 1.5]
```

即 Moderate、Over 样本的分类损失权重提高到 1.5。

#### `four_class_ce_mo_logit_margin_m0.3_lam0.1`

仍是完整四分类。总损失为：

```text
CE + 0.1 × margin_loss
```

只对真实标签为 Moderate/Over 的样本增加边界间隔：

```text
真实 Moderate: relu(0.3 - (logit_moderate - logit_over))
真实 Over:     relu(0.3 - (logit_over - logit_moderate))
```

这里的 `0.3` 是**原始 logit 差值**，不是概率差 30%。当正确边界类相对另一边界类的 logit 差已经达到 0.3，额外惩罚为 0。普通 CE 仍负责区分全部四类。

#### `logistic_normal_cdf_t23_search`

这是专用 EfficientNetV2-S Logistic-Normal 概率序数模型，不与四个普通分类 YAML 做笛卡尔积。前两条边界固定为 `0.25/0.50`，只使用验证集从以下候选值选择 Moderate/Over 边界 `t23`：

```text
0.65, 0.70, 0.75, 0.80, 0.85
```

阈值确定后才在测试集评估一次。绝对不能用测试集选择 `t23`。

### 4. 结果已按模型和实验隔离

每个作业输出到：

```text
temp/moderate_over_boundary_diagnostics/results/
  batch_<timestamp>/
    <model_name>/
      <experiment_name>/
        config.json
        metrics.json
        history.csv
        confusion_matrix.csv
        report.html
```

Logistic-Normal 作业还会生成 `threshold_23_search.csv`。只有显式添加 `--keep-pth` 才会生成 `best_model.pth`。

批次正常走到末尾后会在批次根目录生成：

```text
summary.csv
summary.html
```

单个作业失败时会写 `failure.txt`，随后继续后续作业；只要任一作业失败，全部作业结束后进程退出码为 1。这不代表所有作业都失败。

### 5. CLI 已补齐

支持：

```text
--models
--experiments
--dry-run
--device auto|cuda|cpu
--keep-pth
--list-models
--list-experiments
```

注意：`--models` 只筛选四个 YAML 模型，**不会自动排除独立 Logistic-Normal 作业**。若只想跑一个普通模型实验，必须同时指定 `--experiments`。

### 6. 已做过的验证

- `python -m py_compile temp/moderate_over_boundary_diagnostics/run_experiments.py` 通过。
- 静态作业展开检查为 `4 × 4 + 1 = 17`。
- 曾在 `yolov8` 环境做过代表性 dry-run：四个模型的二分类输出均为 `(2,2)`，四分类输出均为 `(2,4)`，专用 Logistic-Normal 输出为 `(2,4)`，loss 有限。
- dry-run 现在会严格断言输出类别维度，避免二分类模型错误输出 4 类却仍被 CE 判定为可运行。

上述 dry-run 没有持久化日志；如果新会话要把它作为正式依据，应重新运行并保留终端输出。它也不等同于完整训练成功。

## 当前代码与 Git 状态

在创建本交接文件之前：

```text
branch: main
HEAD: 3db95f9872944aa08972b9b10db66bb3097d37d1
origin/main: 3db95f9872944aa08972b9b10db66bb3097d37d1
ahead/behind: 0/0
原工作区: clean
```

多模型脚本和 README 已包含在该提交并已位于远端。`HANDOFF.md` 是本次新建文件，在用户提交前会是新的未提交改动。

此前出现过一次 GitHub 报错：

```text
cannot lock ref 'refs/heads/main': is at 3db95f9... but expected 9cdd0dc...
```

核查后发现本地和远端都已经是 `3db95f9`，实质是并发/重复 push 的引用竞争，目标提交已经到达远端。不要因此使用 `git push --force`。

## 当前卡在哪里

没有已确认的代码 blocker。当前真正停在“代码已完成，但正式结果无法定位或尚未生成”。

审计时：

```text
temp/moderate_over_boundary_diagnostics/results/
```

在当前工作区不存在，因此没有以下内容可供分析：

- batch 目录
- `summary.csv` / `summary.html`
- `metrics.json`
- `failure.txt`
- `best_model.pth`

`temp/moderate_over_boundary_diagnostics/.gitignore` 包含 `results/`，所以即使另一台机器跑出了结果，Git push 也不会把它带过来。用户此前说“刚刚跑的实验”，但仅凭当前工作区无法确认它是否完成、失败、仍在运行，或运行在其他目录/机器。

另一个环境问题是：Windows `(base)` Python 曾报 `ModuleNotFoundError: No module named 'numpy'`。项目可用环境是 `yolov8`；不要在 base 环境下把依赖错误误判为脚本错误。

## 下一步计划

### 第一步：定位用户已有结果

先询问用户实际运行使用的：

- 机器和工作区路径
- PyCharm Run Configuration / 终端命令
- 控制台最后输出
- 实际生成的 `batch_<timestamp>` 目录

如果结果在别处，请复制整个 batch 目录到当前工作区或直接提供其中的 `summary.csv`、`metrics.json`、`failure.txt`。

### 第二步：找不到结果时重新验证环境

在正确环境中运行：

```powershell
conda activate yolov8
python temp/moderate_over_boundary_diagnostics/run_experiments.py --dry-run --device cpu
```

dry-run 强制 `pretrained=False`，不会验证正式运行时的预训练权重下载链路。

### 第三步：先跑单模型、单实验

不要直接用默认右键启动 17 个作业。先跑一个代表性作业，例如：

```powershell
python temp/moderate_over_boundary_diagnostics/run_experiments.py `
  --models resnet50 `
  --experiments four_class_ce_mo_logit_margin_m0.3_lam0.1 `
  --device auto `
  --keep-pth
```

确认该目录中的 `report.html`、`metrics.json`、`history.csv` 和混淆矩阵正常，再扩大到其他模型或实验。

### 第四步：分析结果时重点比较

四分类实验重点看：

- `accuracy`
- `macro_f1`
- `qwk`
- `mae`
- `acc_2_3`
- `error_2_to_3`
- `error_3_to_2`
- Moderate/Over 的逐类 precision、recall、F1
- 完整混淆矩阵

其中 `acc_2_3` 只是筛选真实标签为 2/3 的样本，模型预测仍允许落到 0/1，并不是强制 Moderate/Over 二选一。二分类实验则看 `moderate_over_binary_accuracy`、`moderate_to_over_count` 和 `over_to_moderate_count`。

### 第五步：需要完整批量运行时

正式运行前确认：

- CUDA/GPU 是否可用
- 四个模型的预训练权重能否下载或已缓存
- 磁盘空间是否足够
- 是否真的需要 `--keep-pth`
- 是否能接受 17 个作业、每组最多 150 epoch 的时间成本

如果后续要增强脚本，优先考虑：

1. 每完成一个作业立即增量更新 summary，而不是只在批次末尾写。
2. 保存完整 traceback，而不仅是异常类型和消息。
3. 增加断点续跑/跳过已完成作业能力。

## 数据和训练设置

固定数据规模：

| Split | 总数 | pre | slight | moderate | over | 二分类保留数 |
|---|---:|---:|---:|---:|---:|---:|
| train | 6510 | 1500 | 1500 | 2010 | 1500 | 3510 |
| val | 930 | 210 | 210 | 300 | 210 | 510 |
| test | 1890 | 450 | 450 | 570 | 420 | 990 |

公共训练设置：

| 参数 | 值 |
|---|---:|
| seed | 2026 |
| epochs | 150 |
| batch size | 32 |
| val batch size | 64 |
| test batch size | 64 |
| image size | 224 |
| workers | 4 |
| optimizer | AdamW |
| learning rate | 1e-4 |
| weight decay | 5e-4 |
| warmup | 2 epochs |
| scheduler | cosine |
| minimum LR | 1e-6 |
| early-stopping patience | 30 |

最佳 checkpoint 严格按验证集 Accuracy 的 `>` 更新；Accuracy 平分时保留更早的 epoch。

## 绝对不要再踩的坑

1. **不要声称正式训练已完成。** 当前没有 results、summary 或 metrics 可验证。
2. **不要认为 Git 已推送就等于结果已同步。** `results/` 被 `.gitignore` 忽略。
3. **不要在 base Python 环境运行后把缺 NumPy 当作代码 bug。** 使用项目的 `yolov8` 环境。
4. **不要默认右键只跑一个实验。** 无参数时是 17 个串行作业，每组最多 150 epoch。
5. **不要默认会保存 PTH。** `PYCHARM_KEEP_PTH=False`；需要权重必须加 `--keep-pth`。
6. **不要直接比较二分类 Accuracy 和四分类 Accuracy。** 任务和样本集合不同。
7. **不要把 `acc_2_3` 解释为强制二选一准确率。** 预测仍可能是 pre/slight。
8. **不要把 margin=0.3 解释为概率差 30%。** 它是 raw-logit 间隔。
9. **不要把 Logistic-Normal 头硬套到四个普通 YAML。** MambaOut/SAFNet 的特征与分类头结构不适合直接公平替换；当前专用实验保持独立是有意设计。
10. **绝不能用测试集选择 `t23`。** 只能用验证集搜索，测试集只评估一次。
11. **不要只改 IdentityHead 的类别数。** MambaOut/SAFNet 二分类必须改 `backbone.num_classes`。
12. **不要直接把四个模型 YAML 当完整 TrainingConfig。** 它们只有 `name/model` 片段，缺少 data/train/optimizer 等公共段。
13. **不要在模型构建时复用会被修改的原始字典。** `ImageClassifier` 会 `pop("type")`，必须 deepcopy。
14. **不要盲信 MambaOut 名称。** 请求模型不存在时 wrapper 可能 fallback；检查 `metrics.json` 中的 `requested_backbone_model_name` 与 `actual_backbone_model_name`。
15. **不要把 SAFNet 的 `pretrained=true` 理解为完整 SAFNet 预训练。** 它只部分加载 ImageNet ResNet50 兼容权重，新增 CCA/SE/AMSAFF 层仍是随机初始化。
16. **不要把最终退出码 1 理解为全部失败。** 脚本会记录单个失败并继续，结束时只要有一个失败就返回 1。
17. **不要假设中断批次一定有 summary。** summary 只在主循环末尾写；中断后可能只有若干子目录。
18. **不要用 `git push --force` 解决此前的 ref-lock 报错。** 先比较本地 HEAD、`origin/main` 和 `git ls-remote`。
19. **不要为了这个临时诊断修改正式 YAML、`tools/train_batch.py` 或原 `runs/`。** 临时产物应保持在诊断目录内。

