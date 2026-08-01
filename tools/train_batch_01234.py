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

# 01234 五分类 BaSiC 预处理 + 固定网格 30 patch、408×408 输入专用公共配置。
train_batch_base.COMMON_CONFIG = Path('configs/fixed_split_01234_BaSic_grid30_408_train.yaml')

# 本次队列：把 temp 里效果较好的 MixNet-S 结构搜索配置抽到固定目录后复验。
# 公共数据集、epoch、batch、optimizer、scheduler、loss 和权重清理策略都继承 01234 BaSiC/grid30/408 公共配置。
train_batch_base.CONFIG_LIST = (
    Path('configs/fixed_split_01234_models/mixnet_structure_selected/stagemask_001101_k357.yaml'),
    Path('configs/fixed_split_01234_models/mixnet_structure_selected/10_p10_only_s0_k357.yaml'),
    Path('configs/fixed_split_01234_models/mixnet_structure_selected/stagemask_000101_k357.yaml'),
    Path('configs/fixed_split_01234_models/mixnet_structure_selected/p03_stride2_k357_g3_softmax.yaml'),
    Path('configs/fixed_split_01234_models/mixnet_structure_selected/stagemask_001001_k357.yaml'),
    Path('configs/fixed_split_01234_models/mixnet_structure_selected/13_p13_only_s3_k357.yaml'),
    Path('configs/fixed_split_01234_models/mixnet_structure_selected/stagemask_011010_k357.yaml'),
    Path('configs/fixed_split_01234_models/mixnet_structure_selected/12_p12_only_s2_k357.yaml'),
    Path('configs/fixed_split_01234_models/mixnet_structure_selected/09_p09_late_s345_k357.yaml'),
    Path('configs/fixed_split_01234_models/mixnet_structure_selected/stagemask_111101_k357.yaml'),
    Path('configs/fixed_split_01234_models/mixnet_structure_selected/p08_midlate_s2345_k3579.yaml'),
    Path('configs/fixed_split_01234_models/mixnet_structure_selected/p03_stride2_k3579.yaml'),
    Path('configs/fixed_split_01234_models/mixnet_structure_selected/16_p16_first_block_k357.yaml'),
    Path('configs/fixed_split_01234_models/mixnet_structure_selected/stagemask_111100_k357.yaml'),
    Path('configs/fixed_split_01234_models/mixnet_structure_selected/p07_last2_s45_k35.yaml'),
)

# PyCharm 右键运行默认设置；也可以继续用命令行参数覆盖。
train_batch_base.PYCHARM_DEVICE = 'auto'
train_batch_base.PYCHARM_DRY_RUN = False
train_batch_base.PYCHARM_FAIL_FAST = False
train_batch_base.PYCHARM_KEEP_PTH_FILES = False


def main() -> None:
    """进入原 train_batch.py 的主流程，只是换成 01234 专用配置列表。"""
    # 这批复验明确不长期保留权重，防止命令行误传 --keep-pth 覆盖公共配置。
    if any(arg in {'--keep-pth', '--keep-pth-files'} for arg in sys.argv[1:]):
        raise SystemExit('本批 01234 复验不保存 .pth 权重文件，请不要传 --keep-pth。')
    train_batch_base.main()


if __name__ == '__main__':
    main()
