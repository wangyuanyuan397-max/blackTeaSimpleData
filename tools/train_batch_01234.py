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

# 01234 五分类固定网格 30 patch、408×408 输入专用公共配置。
train_batch_base.COMMON_CONFIG = Path('configs/fixed_split_01234_grid30_408_train.yaml')

# 本次先跑 4 个 torchvision baseline，再追加本地 timm 已确认支持的轻量模型。
train_batch_base.CONFIG_LIST = (
    Path('configs/fixed_split_01234_models/fixed_mobilenet_v3_small.yaml'),
    Path('configs/fixed_split_01234_models/fixed_mobilenet_v3_large.yaml'),
    Path('configs/fixed_split_01234_models/fixed_efficientnet_b0.yaml'),
    Path('configs/fixed_split_01234_models/fixed_efficientnet_v2_s.yaml'),
    Path('configs/fixed_split_01234_models/fixed_timm_mobilenetv1_100.yaml'),
    Path('configs/fixed_split_01234_models/fixed_timm_mobilenetv2_100.yaml'),
    Path('configs/fixed_split_01234_models/fixed_timm_mnasnet_100.yaml'),
    Path('configs/fixed_split_01234_models/fixed_timm_mobilenetv3_small_100.yaml'),
    Path('configs/fixed_split_01234_models/fixed_timm_mobilenetv3_large_100.yaml'),
    Path('configs/fixed_split_01234_models/fixed_timm_efficientnet_b0.yaml'),
    Path('configs/fixed_split_01234_models/fixed_timm_efficientnet_b1.yaml'),
    Path('configs/fixed_split_01234_models/fixed_timm_mixnet_s.yaml'),
    Path('configs/fixed_split_01234_models/fixed_timm_mixnet_m.yaml'),
    Path('configs/fixed_split_01234_models/fixed_timm_mixnet_l.yaml'),
    Path('configs/fixed_split_01234_models/fixed_timm_ghostnet_100.yaml'),
    Path('configs/fixed_split_01234_models/fixed_timm_ghostnetv2_100.yaml'),
    Path('configs/fixed_split_01234_models/fixed_timm_tf_efficientnetv2_s.yaml'),
    Path('configs/fixed_split_01234_models/fixed_timm_mobilevit_xxs.yaml'),
    Path('configs/fixed_split_01234_models/fixed_timm_mobilevit_xs.yaml'),
    Path('configs/fixed_split_01234_models/fixed_timm_mobilevit_s.yaml'),
    Path('configs/fixed_split_01234_models/fixed_timm_edgenext_xx_small.yaml'),
    Path('configs/fixed_split_01234_models/fixed_timm_edgenext_small.yaml'),
    Path('configs/fixed_split_01234_models/fixed_timm_efficientformer_l1.yaml'),
    Path('configs/fixed_split_01234_models/fixed_timm_efficientformerv2_s0.yaml'),
    Path('configs/fixed_split_01234_models/fixed_timm_mobileone_s0.yaml'),
    Path('configs/fixed_split_01234_models/fixed_timm_mobileone_s1.yaml'),
    Path('configs/fixed_split_01234_models/fixed_timm_mobileone_s2.yaml'),
    Path('configs/fixed_split_01234_models/fixed_timm_mobileone_s3.yaml'),
    Path('configs/fixed_split_01234_models/fixed_timm_mobileone_s4.yaml'),
    Path('configs/fixed_split_01234_models/fixed_timm_fasternet_t0.yaml'),
    Path('configs/fixed_split_01234_models/fixed_timm_fasternet_t1.yaml'),
    Path('configs/fixed_split_01234_models/fixed_timm_fasternet_t2.yaml'),
    Path('configs/fixed_split_01234_models/fixed_timm_repvit_m0_9.yaml'),
    Path('configs/fixed_split_01234_models/fixed_timm_repvit_m1_0.yaml'),
    Path('configs/fixed_split_01234_models/fixed_timm_repvit_m1_1.yaml'),
    Path('configs/fixed_split_01234_models/fixed_timm_mobilenetv4_conv_small.yaml'),
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
