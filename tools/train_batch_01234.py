"""批量运行 datasets_01234 五分类实验。

这个脚本复用 tools/train_batch.py 的训练、评估、HTML 报告和 pth 清理逻辑，
但使用独立的 5 类公共配置和 5 类模型 YAML，避免污染原来的四分类实验入口。
"""

import importlib.util
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE_TRAIN_BATCH_PATH = PROJECT_ROOT / 'tools' / 'train_batch.py'

spec = importlib.util.spec_from_file_location('train_batch_base', BASE_TRAIN_BATCH_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(f'无法加载基础批训练脚本：{BASE_TRAIN_BATCH_PATH}')

train_batch_base = importlib.util.module_from_spec(spec)
sys.modules['train_batch_base'] = train_batch_base
spec.loader.exec_module(train_batch_base)

# 01234 五分类专用公共配置。
train_batch_base.COMMON_CONFIG = Path('configs/fixed_split_01234_train.yaml')

# 本次要跑的全部模型：先 fixed_split_patches_models 下 13 个，再 tryPractice/Attention 下 31 个。
train_batch_base.CONFIG_LIST = (
    Path('configs/fixed_split_01234_models/fixed_convnext_tiny.yaml'),
    Path('configs/fixed_split_01234_models/fixed_efficientnet_v2_s.yaml'),
    Path('configs/fixed_split_01234_models/fixed_efficientnet_v2_s_ablation_conv_refine.yaml'),
    Path('configs/fixed_split_01234_models/fixed_efficientnet_v2_s_ablation_linear.yaml'),
    Path('configs/fixed_split_01234_models/fixed_efficientnet_v2_s_ablation_mlp.yaml'),
    Path('configs/fixed_split_01234_models/fixed_efficientnet_v2_s_ablation_se_refine.yaml'),
    Path('configs/fixed_split_01234_models/fixed_efficientnet_v2_s_gated_refinement.yaml'),
    Path('configs/fixed_split_01234_models/fixed_efficientnet_v2_s_multistage_gated_fusion.yaml'),
    Path('configs/fixed_split_01234_models/fixed_mambaout_tiny.yaml'),
    Path('configs/fixed_split_01234_models/fixed_mobilenet_v3_large.yaml'),
    Path('configs/fixed_split_01234_models/fixed_resnet50.yaml'),
    Path('configs/fixed_split_01234_models/fixed_safnet_imagenet.yaml'),
    Path('configs/fixed_split_01234_models/fixed_safnet_scratch.yaml'),
    Path('configs/fixed_split_01234_models/attention_baseline.yaml'),
    Path('configs/fixed_split_01234_models/attention_ca_p1.yaml'),
    Path('configs/fixed_split_01234_models/attention_ca_p2.yaml'),
    Path('configs/fixed_split_01234_models/attention_ca_p3.yaml'),
    Path('configs/fixed_split_01234_models/attention_ca_p4.yaml'),
    Path('configs/fixed_split_01234_models/attention_ca_p456.yaml'),
    Path('configs/fixed_split_01234_models/attention_ca_p46.yaml'),
    Path('configs/fixed_split_01234_models/attention_ca_p5.yaml'),
    Path('configs/fixed_split_01234_models/attention_ca_p56.yaml'),
    Path('configs/fixed_split_01234_models/attention_ca_p6.yaml'),
    Path('configs/fixed_split_01234_models/attention_ca_pall.yaml'),
    Path('configs/fixed_split_01234_models/attention_cbam_p1.yaml'),
    Path('configs/fixed_split_01234_models/attention_cbam_p2.yaml'),
    Path('configs/fixed_split_01234_models/attention_cbam_p3.yaml'),
    Path('configs/fixed_split_01234_models/attention_cbam_p4.yaml'),
    Path('configs/fixed_split_01234_models/attention_cbam_p456.yaml'),
    Path('configs/fixed_split_01234_models/attention_cbam_p46.yaml'),
    Path('configs/fixed_split_01234_models/attention_cbam_p5.yaml'),
    Path('configs/fixed_split_01234_models/attention_cbam_p56.yaml'),
    Path('configs/fixed_split_01234_models/attention_cbam_p6.yaml'),
    Path('configs/fixed_split_01234_models/attention_cbam_pall.yaml'),
    Path('configs/fixed_split_01234_models/attention_eca_p1.yaml'),
    Path('configs/fixed_split_01234_models/attention_eca_p2.yaml'),
    Path('configs/fixed_split_01234_models/attention_eca_p3.yaml'),
    Path('configs/fixed_split_01234_models/attention_eca_p4.yaml'),
    Path('configs/fixed_split_01234_models/attention_eca_p456.yaml'),
    Path('configs/fixed_split_01234_models/attention_eca_p46.yaml'),
    Path('configs/fixed_split_01234_models/attention_eca_p5.yaml'),
    Path('configs/fixed_split_01234_models/attention_eca_p56.yaml'),
    Path('configs/fixed_split_01234_models/attention_eca_p6.yaml'),
    Path('configs/fixed_split_01234_models/attention_eca_pall.yaml'),
)

# PyCharm 右键运行默认设置；也可以继续用命令行参数覆盖。
train_batch_base.PYCHARM_DEVICE = 'auto'
train_batch_base.PYCHARM_DRY_RUN = False
train_batch_base.PYCHARM_FAIL_FAST = False
train_batch_base.PYCHARM_KEEP_PTH_FILES = False


def main() -> None:
    """进入原 train_batch.py 的主流程，只是换成 01234 专用配置列表。"""
    train_batch_base.main()


if __name__ == '__main__':
    main()
