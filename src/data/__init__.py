"""固定目录图像数据集及其数据变换注册入口。"""

from .loader import ImageFolderWithPaths, build_dataloader

__all__ = ["ImageFolderWithPaths", "build_dataloader"]
