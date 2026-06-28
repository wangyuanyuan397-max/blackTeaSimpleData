"""精简训练仓库中日志与报告共同使用的基础常量。"""

from typing import List


APP_NAME = "black-tea-simple-data"
DEFAULT_CLASS_ORDER = ("pre", "slight", "moderate", "over")


def get_class_order() -> List[str]:
    """返回与固定数据集 class_to_idx 一致的类别顺序副本。"""
    return list(DEFAULT_CLASS_ORDER)
