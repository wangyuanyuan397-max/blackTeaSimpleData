"""Adapter for the official HUST-VL Vim implementation."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from ...utils.registry import BACKBONES


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SUPPORTED_CHECKPOINT_SUFFIXES = (".pth", ".pt", ".bin", ".safetensors")


def _resolve_project_path(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return value if value.is_absolute() else PROJECT_ROOT / value


def _candidate_model_files(code_dir: Optional[str]) -> list[Path]:
    candidates: list[Path] = []
    if code_dir:
        root = _resolve_project_path(code_dir)
        candidates.extend([root / "models_mamba.py", root / "vim" / "models_mamba.py"])
    candidates.extend(
        [
            PROJECT_ROOT / "vim" / "models_mamba.py",
            PROJECT_ROOT / "external" / "Vim" / "vim" / "models_mamba.py",
            PROJECT_ROOT / "third_party" / "Vim" / "vim" / "models_mamba.py",
        ]
    )
    unique_candidates: list[Path] = []
    seen = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved not in seen:
            unique_candidates.append(candidate)
            seen.add(resolved)
    return unique_candidates


def _import_official_vim_module(code_dir: Optional[str]):
    candidates = _candidate_model_files(code_dir)
    model_file = next((path for path in candidates if path.is_file()), None)
    if model_file is None:
        searched = "\n".join(f"  - {path}" for path in candidates)
        raise FileNotFoundError(
            "未找到官方 Vim 结构文件 models_mamba.py。请把 HUST-VL/Vim 官方仓库放到 "
            "external/Vim，或在 YAML 的 code_dir 指向包含 vim/models_mamba.py 的目录。\n"
            f"已搜索：\n{searched}"
        )

    module_dir = str(model_file.parent)
    repo_root = str(model_file.parent.parent)
    for path in (repo_root, module_dir):
        if path not in sys.path:
            sys.path.insert(0, path)

    spec = importlib.util.spec_from_file_location("_external_hustvl_vim_models_mamba", model_file)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法导入 Vim 结构文件：{model_file}")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            f"导入 Vim 官方代码失败，缺少依赖：{exc.name}。通常需要安装 mamba_ssm、causal_conv1d、triton，"
            "并保证官方 Vim 仓库中的 vim/ 目录完整。"
        ) from exc
    return module


def _extract_state_dict(checkpoint: Any) -> Dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict):
        for key in ("model", "state_dict", "model_state_dict"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                checkpoint = checkpoint[key]
                break
    if not isinstance(checkpoint, dict):
        raise TypeError("checkpoint 中没有可用的 state_dict。")
    return dict(checkpoint)


def _load_checkpoint_file(checkpoint_path: Path) -> Dict[str, torch.Tensor]:
    if checkpoint_path.suffix == ".safetensors":
        from safetensors.torch import load_file

        checkpoint = load_file(str(checkpoint_path))
    else:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    return _extract_state_dict(checkpoint)


def _find_checkpoint_in_snapshot(snapshot_dir: Path) -> Path:
    files = [
        path
        for path in snapshot_dir.rglob("*")
        if path.is_file() and path.suffix in SUPPORTED_CHECKPOINT_SUFFIXES
    ]
    if not files:
        raise FileNotFoundError(f"Hugging Face snapshot 中没有找到权重文件：{snapshot_dir}")
    files.sort(key=lambda path: ("model" not in path.name.lower(), len(path.name), path.name))
    return files[0]


def _resolve_checkpoint_path(
    checkpoint_path: Optional[str],
    pretrained_repo: str,
    checkpoint_filename: Optional[str],
) -> Path:
    if checkpoint_path:
        path = _resolve_project_path(checkpoint_path)
        if not path.is_file():
            raise FileNotFoundError(f"Vim checkpoint 不存在：{path}")
        return path

    try:
        from huggingface_hub import hf_hub_download, snapshot_download
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("加载 Vim 预训练权重需要安装 huggingface_hub。") from exc

    if checkpoint_filename:
        return Path(hf_hub_download(repo_id=pretrained_repo, filename=checkpoint_filename))
    snapshot_dir = Path(snapshot_download(repo_id=pretrained_repo))
    return _find_checkpoint_in_snapshot(snapshot_dir)


def _load_pretrained_without_head(model: nn.Module, checkpoint_path: Path) -> None:
    model_state = model.state_dict()
    raw_state = _load_checkpoint_file(checkpoint_path)
    compatible_state: Dict[str, torch.Tensor] = {}
    removed_keys: list[str] = []

    for raw_key, value in raw_state.items():
        key = raw_key
        for prefix in ("module.", "model."):
            stripped = key[len(prefix):] if key.startswith(prefix) else key
            if stripped in model_state:
                key = stripped
                break
        if key not in model_state or model_state[key].shape != value.shape:
            removed_keys.append(raw_key)
            continue
        compatible_state[key] = value

    incompatible = model.load_state_dict(compatible_state, strict=False)
    if len(compatible_state) == 0:
        raise RuntimeError(f"没有任何 Vim 预训练权重成功匹配当前模型：{checkpoint_path}")
    print(f"Vim pretrained loaded: {checkpoint_path}")
    if removed_keys:
        print(f"Vim removed mismatched/unexpected keys: {len(removed_keys)}")
    if incompatible.missing_keys:
        print(f"Vim missing keys: {len(incompatible.missing_keys)}")
    if incompatible.unexpected_keys:
        print(f"Vim unexpected keys: {len(incompatible.unexpected_keys)}")


@BACKBONES.register("vim_small_midclstok")
class VimSmallMiddleClsTokenBackbone(nn.Module):
    """Vim-Small Middle-CLS model that returns 4-class logits directly."""

    def __init__(
        self,
        num_classes: int = 4,
        pretrained: bool = True,
        code_dir: Optional[str] = "external/Vim",
        pretrained_repo: str = "hustvl/Vim-small-midclstok",
        checkpoint_path: Optional[str] = None,
        checkpoint_filename: Optional[str] = None,
        model_builder: str = "vim_small_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2",
        **model_kwargs,
    ):
        super().__init__()
        module = _import_official_vim_module(code_dir)
        if not hasattr(module, model_builder):
            raise AttributeError(f"models_mamba.py 中没有找到 Vim 构造函数：{model_builder}")
        build_model = getattr(module, model_builder)
        self.model = build_model(pretrained=False, num_classes=num_classes, **model_kwargs)
        if pretrained:
            resolved_checkpoint = _resolve_checkpoint_path(
                checkpoint_path=checkpoint_path,
                pretrained_repo=pretrained_repo,
                checkpoint_filename=checkpoint_filename,
            )
            _load_pretrained_without_head(self.model, resolved_checkpoint)
        self.out_features = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
