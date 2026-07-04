'''递归查找并删除指定目录下的所有 .pth 文件。'''

import argparse
from pathlib import Path
from typing import Iterable


# ============================== PyCharm 右键运行配置区 ==============================
# 将这里改成需要清理的目录；脚本会继续搜索它下面的所有多级子目录。
TARGET_DIRECTORY = Path('G:/wyy/projects/blackTeaSimpleData/runs')

# True 只显示将要删除的文件，不执行删除；确认无误后改成 False。
DRY_RUN = False


def format_size(size_bytes: int) -> str:
    '''把字节数转换成便于阅读的 KB、MB、GB。'''
    size = float(size_bytes)
    for unit in ('B', 'KB', 'MB', 'GB', 'TB'):
        if size < 1024.0 or unit == 'TB':
            return f'{size:.2f} {unit}'
        size /= 1024.0
    return f'{size_bytes} B'


def validate_target_directory(directory: Path) -> Path:
    '''解析并检查目标目录，拒绝磁盘根目录等过于宽泛的危险路径。'''
    resolved = directory.expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f'目标目录不存在：{resolved}')
    if not resolved.is_dir():
        raise NotADirectoryError(f'目标路径不是目录：{resolved}')
    if resolved == Path(resolved.anchor):
        raise ValueError(f'拒绝直接清理磁盘根目录：{resolved}')
    return resolved


def find_pth_files(directory: Path) -> list[Path]:
    '''递归返回所有扩展名为 .pth 的普通文件，并按完整路径排序。'''
    return sorted(
        (
            path
            for path in directory.rglob('*')
            if path.is_file() and path.suffix.lower() == '.pth'
        ),
        key=lambda path: str(path).lower(),
    )


def total_file_size(paths: Iterable[Path]) -> int:
    '''统计文件总大小；若文件在统计期间消失，则忽略该文件。'''
    total = 0
    for path in paths:
        try:
            total += path.stat().st_size
        except FileNotFoundError:
            continue
    return total


def delete_pth_files(paths: Iterable[Path]) -> tuple[list[Path], list[tuple[Path, str]]]:
    '''逐个删除文件，返回成功列表和失败原因，不因单个文件失败而中断。'''
    deleted = []
    failed = []
    for path in paths:
        try:
            path.unlink()
            deleted.append(path)
        except FileNotFoundError:
            continue
        except OSError as error:
            failed.append((path, str(error)))
    return deleted, failed


def parse_arguments() -> argparse.Namespace:
    '''解析命令行参数；未传目录时使用顶部 TARGET_DIRECTORY。'''
    parser = argparse.ArgumentParser(
        description='递归预览或删除指定目录下的全部 .pth 文件。'
    )
    parser.add_argument(
        'directory',
        nargs='?',
        type=Path,
        default=TARGET_DIRECTORY,
        help=f'目标目录；默认使用代码顶部路径：{TARGET_DIRECTORY}',
    )
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        '--execute',
        action='store_true',
        help='真正删除找到的 .pth；不传时默认只预览。',
    )
    mode_group.add_argument(
        '--dry-run',
        action='store_true',
        help='强制只预览，即使代码顶部 DRY_RUN=False。',
    )
    return parser.parse_args()


def main() -> None:
    '''执行递归扫描，并根据开关进行预览或删除。'''
    args = parse_arguments()
    target_directory = validate_target_directory(args.directory)
    dry_run = True if args.dry_run else (False if args.execute else DRY_RUN)
    pth_files = find_pth_files(target_directory)
    total_size = total_file_size(pth_files)

    print(f'目标目录：{target_directory}')
    print(f'找到 .pth：{len(pth_files)} 个')
    print(f'占用空间：{format_size(total_size)}')

    if not pth_files:
        print('没有需要处理的 .pth 文件。')
        return

    for path in pth_files:
        try:
            size_text = format_size(path.stat().st_size)
        except FileNotFoundError:
            size_text = '文件已不存在'
        print(f'  - {path} ({size_text})')

    if dry_run:
        print()
        print('当前为预览模式，没有删除任何文件。')
        print('确认路径无误后：')
        print('1. PyCharm 右键运行：将代码顶部 DRY_RUN 改为 False；')
        print('2. 命令行运行：添加 --execute。')
        return

    deleted, failed = delete_pth_files(pth_files)
    failed_size = total_file_size(path for path, _ in failed)
    freed_size = max(total_size - failed_size, 0)
    print()
    print(f'删除成功：{len(deleted)} 个')
    print(f'预计释放：{format_size(freed_size)}')
    if failed:
        print(f'删除失败：{len(failed)} 个')
        for path, reason in failed:
            print(f'  - {path}: {reason}')
        raise SystemExit(1)


if __name__ == '__main__':
    main()
