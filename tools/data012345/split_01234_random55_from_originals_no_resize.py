"""生成不进行 resize 的 datasets_01234 五分类随机裁剪数据集。

这个脚本复用 split_01234_random55_from_originals.py 的全部划分和裁剪逻辑，
唯一差异是：裁剪出 408x408 patch 后直接保存，不缩放到 224x224。

右键运行本文件即可。输出目录默认是：datasets_01234_408。
"""

from pathlib import Path

import split_01234_random55_from_originals as base


# 不覆盖原来的 224x224 数据集，单独输出 408x408 版本。
base.OUTPUT_ROOT = Path(r"E:\workspaces\python\BlackTeaSimpleData\datasets_01234_408")

# 关闭 resize：保存原始 408x408 裁剪块。
base.ENABLE_RESIZE = False


if __name__ == "__main__":
    base.main()
