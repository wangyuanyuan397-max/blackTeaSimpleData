"""
薄实例配置编译器。

职责：
- 读取 recipe / protocol / runtime / overrides
- 使用 OmegaConf.merge(...) 合并配置层
- 为调试脚本提供编译 bundle 和差异分析基础
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

try:
    from omegaconf import OmegaConf
except ImportError:  # pragma: no cover - import side effect
    OmegaConf = None


PROJECT_ROOT = Path(__file__).resolve().parents[2]
THIN_RUN_KEYS = {"recipe", "protocol", "runtime", "overrides"}


class ConfigCompilerError(RuntimeError):
    """配置编译失败时抛出的统一错误。"""


@dataclass(frozen=True)
class CompiledRunBundle:
    """保存一次薄实例编译所需的全部上下文。"""

    run_path: Path
    run_data: dict[str, Any]
    recipe_path: Path
    recipe_data: dict[str, Any]
    protocol_path: Path
    protocol_data: dict[str, Any]
    runtime_path: Path | None
    runtime_data: dict[str, Any]
    overrides: dict[str, Any]
    compiled_data: dict[str, Any]


def ensure_omegaconf_available() -> None:
    """确保运行时存在 OmegaConf。"""

    if OmegaConf is None:
        raise ConfigCompilerError(
            "OmegaConf is required for thin-run compilation. "
            "Please install it first, e.g. `pip install omegaconf`."
        )


def load_yaml_raw(path: str | Path) -> dict[str, Any]:
    """读取 YAML 并返回顶层 dict。"""

    yaml_path = Path(path)
    if not yaml_path.exists():
        raise FileNotFoundError(f"YAML file not found: {yaml_path}")

    with yaml_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    if not isinstance(data, dict):
        raise ConfigCompilerError(f"Expected top-level mapping in YAML: {yaml_path}")

    return data


def is_thin_run_config(data: dict[str, Any]) -> bool:
    """判断一个 YAML 是否是薄实例格式。"""

    return isinstance(data, dict) and "recipe" in data and "protocol" in data


def resolve_config_reference(
    reference: str | Path,
    *,
    relative_to: Path | None = None,
    project_root: Path = PROJECT_ROOT,
) -> Path:
    """解析薄实例里引用的 recipe/protocol/runtime 路径。"""

    ref_path = Path(reference)
    candidates: list[Path] = []

    if ref_path.is_absolute():
        candidates.append(ref_path)
    else:
        if relative_to is not None:
            candidates.append((relative_to / ref_path).resolve())
        candidates.append((project_root / ref_path).resolve())
        candidates.append(ref_path.resolve())

    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(f"Config reference not found: {reference}")


def merge_config_layers(
    recipe: dict[str, Any],
    protocol: dict[str, Any] | None = None,
    runtime: dict[str, Any] | None = None,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """按 recipe < protocol < runtime < overrides 的顺序合并配置。"""

    ensure_omegaconf_available()
    layers = [recipe or {}, protocol or {}, runtime or {}, overrides or {}]
    merged = OmegaConf.merge(*(OmegaConf.create(layer) for layer in layers))
    data = OmegaConf.to_container(merged, resolve=True)
    if not isinstance(data, dict):
        raise ConfigCompilerError("Merged configuration must be a mapping")
    return data


def build_compilation_bundle(run_path: str | Path) -> CompiledRunBundle:
    """读取薄实例并构建完整编译上下文。"""

    run_file = Path(run_path).resolve()
    run_data = load_yaml_raw(run_file)

    if not is_thin_run_config(run_data):
        raise ConfigCompilerError(
            f"Not a thin run config: {run_file}. Expected keys: recipe + protocol."
        )

    unknown_keys = sorted(set(run_data) - THIN_RUN_KEYS)
    if unknown_keys:
        raise ConfigCompilerError(
            f"Thin run config contains unsupported top-level keys: {unknown_keys}. "
            "Please keep full experiment semantics in recipe and put local changes under overrides."
        )

    overrides = run_data.get("overrides") or {}
    if not isinstance(overrides, dict):
        raise ConfigCompilerError(f"`overrides` must be a mapping: {run_file}")

    recipe_path = resolve_config_reference(run_data["recipe"], relative_to=run_file.parent)
    protocol_path = resolve_config_reference(run_data["protocol"], relative_to=run_file.parent)
    runtime_ref = run_data.get("runtime")
    runtime_path = (
        resolve_config_reference(runtime_ref, relative_to=run_file.parent) if runtime_ref else None
    )

    recipe_data = load_yaml_raw(recipe_path)
    protocol_data = load_yaml_raw(protocol_path)
    runtime_data = load_yaml_raw(runtime_path) if runtime_path else {}
    compiled_data = merge_config_layers(recipe_data, protocol_data, runtime_data, overrides)

    return CompiledRunBundle(
        run_path=run_file,
        run_data=run_data,
        recipe_path=recipe_path,
        recipe_data=recipe_data,
        protocol_path=protocol_path,
        protocol_data=protocol_data,
        runtime_path=runtime_path,
        runtime_data=runtime_data,
        overrides=overrides,
        compiled_data=compiled_data,
    )


def compile_run_config(run_path: str | Path) -> dict[str, Any]:
    """编译薄实例并返回最终完整配置。"""

    return build_compilation_bundle(run_path).compiled_data


def materialize_run_config(run_path: str | Path, output_path: str | Path) -> None:
    """将编译后的完整配置落盘为 YAML。"""

    compiled = compile_run_config(run_path)
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as f:
        yaml.safe_dump(compiled, f, allow_unicode=True, sort_keys=False)


def diff_values(base: Any, current: Any, prefix: str = "") -> list[tuple[str, Any, Any]]:
    """递归比较两个对象，返回变更路径。"""

    changes: list[tuple[str, Any, Any]] = []
    path_label = prefix or "<root>"

    if isinstance(base, dict) and isinstance(current, dict):
        for key in sorted(set(base) | set(current)):
            next_prefix = f"{prefix}.{key}" if prefix else str(key)
            if key not in base:
                changes.append((next_prefix, "<missing>", current[key]))
                continue
            if key not in current:
                changes.append((next_prefix, base[key], "<missing>"))
                continue
            changes.extend(diff_values(base[key], current[key], next_prefix))
        return changes

    if isinstance(base, list) and isinstance(current, list):
        if base != current:
            changes.append((path_label, base, current))
        return changes

    if base != current:
        changes.append((path_label, base, current))
    return changes
