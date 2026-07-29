"""Generate AD-GBC brute-force YAML configs."""

from __future__ import annotations

import argparse
from itertools import product
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
CONFIG_ROOT = THIS_DIR / "configs"

MODEL_NAME = "mixnet_s"
SEED = 2026
ADGBC_DIM = 256
# 暴力搜索网格：覆盖 K、tau、两个几何损失权重和不同训练策略。
K_VALUES = (8, 16, 32, 64)
TAU_VALUES = (1.0, 0.5, 0.1)
LAMBDA_W_VALUES = (0.0, 0.01, 0.05, 0.1)
BETA_SCALE_VALUES = (0.0, 0.05, 0.1)
TRAINING_STRATEGIES = (
    "finetune_all",
    "warmup5_then_finetune",
    "adgbc_head_only",
    "head_only",
)

SMOKE_GRID = {
    "k_values": (16,),
    "tau_values": (1.0, 0.5),
    "lambda_w_values": (0.0, 0.05),
    "beta_scale_values": (0.0, 0.05),
    "training_strategies": ("finetune_all", "adgbc_head_only"),
}


def slug_float(value: float) -> str:
    text = f"{float(value):g}"
    return text.replace("-", "m").replace(".", "p")


def strategy_overrides(strategy: str) -> tuple[list[str], list[str]]:
    backbone_lines: list[str] = []
    train_lines: list[str] = []
    if strategy == "finetune_all":
        # 全量端到端微调：backbone、AD-GBC、分类头一起训练。
        return backbone_lines, train_lines
    if strategy == "adgbc_head_only":
        # 冻结原始 backbone，只训练新增 AD-GBC 净化器和分类头。
        backbone_lines.append("    freeze_base: true")
        return backbone_lines, train_lines
    if strategy == "head_only":
        # 负对照：冻结 backbone 和 AD-GBC，只训练最后分类头。
        backbone_lines.extend(
            [
                "    freeze_base: true",
                "    freeze_adgbc: true",
            ]
        )
        return backbone_lines, train_lines
    if strategy == "warmup5_then_finetune":
        # 先让 AD-GBC/分类头适应 5 个 epoch，再解冻 backbone 端到端微调。
        train_lines.extend(
            [
                "train:",
                "  staged_training:",
                "    enabled: true",
                "    freeze_backbone_epochs: 5",
                "    stage1_lr: 0.0001",
                "    stage2_lr: 0.0001",
                "    stage1_warmup_epochs: 2",
                "    stage2_warmup_epochs: 2",
            ]
        )
        return backbone_lines, train_lines
    raise ValueError(f"Unknown training strategy: {strategy}")


def config_name(
    k: int,
    tau: float,
    lambda_w: float,
    beta_scale: float,
    strategy: str,
) -> str:
    return (
        f"adgbc_{MODEL_NAME}"
        f"_k{k}"
        f"_tau{slug_float(tau)}"
        f"_lw{slug_float(lambda_w)}"
        f"_bs{slug_float(beta_scale)}"
        f"_{strategy}"
    )


def write_config(
    path: Path,
    *,
    k: int,
    tau: float,
    lambda_w: float,
    beta_scale: float,
    strategy: str,
) -> None:
    name = path.stem
    backbone_extra, train_extra = strategy_overrides(strategy)
    lines = [
        "# AD-GBC 净化器插在 timm backbone 的 feature map 和全局池化之间。",
        f"name: {name}",
        f"random_seed: {SEED}",
        "model:",
        "  type: classifier",
        "  strategy: classification",
        "  backbone:",
        "    type: timm_adgbc",
        f"    model_name: {MODEL_NAME}",
        "    pretrained: true",
        "    input_size: 408",
        "    # adgbc_k 是弹性区域数量；adgbc_tau 越小，软分配越尖锐。",
        f"    adgbc_k: {int(k)}",
        f"    adgbc_dim: {int(ADGBC_DIM)}",
        f"    adgbc_tau: {float(tau):g}",
        "    normalize_region_desc: true",
        "    use_refine: true",
        *backbone_extra,
        "  head:",
        "    type: linear",
        "    drop_rate: 0.0",
        "loss:",
        "  type: adgbc_cross_entropy",
        "  label_smoothing: 0.0",
        "  # lambda_w_div 控制区域多样性；beta_scale_con 控制尺度一致性。",
        f"  lambda_w_div: {float(lambda_w):g}",
        f"  beta_scale_con: {float(beta_scale):g}",
    ]
    lines.extend(train_extra)
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def write_baseline_config(path: Path) -> None:
    lines = [
        "# 原始 MixNet-S baseline，用来和 AD-GBC 暴力搜索结果对照。",
        f"name: {path.stem}",
        f"random_seed: {SEED}",
        "model:",
        "  type: classifier",
        "  strategy: classification",
        "  backbone:",
        "    type: timm",
        f"    model_name: {MODEL_NAME}",
        "    pretrained: true",
        "    input_size: 408",
        "  head:",
        "    type: linear",
        "    drop_rate: 0.0",
        "loss:",
        "  type: cross_entropy",
        "  label_smoothing: 0.0",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def generate_grid(
    directory: Path,
    *,
    k_values: tuple[int, ...],
    tau_values: tuple[float, ...],
    lambda_w_values: tuple[float, ...],
    beta_scale_values: tuple[float, ...],
    training_strategies: tuple[str, ...],
) -> list[Path]:
    generated: list[Path] = []
    for k, tau, lambda_w, beta_scale, strategy in product(
        k_values,
        tau_values,
        lambda_w_values,
        beta_scale_values,
        training_strategies,
    ):
        name = config_name(k, tau, lambda_w, beta_scale, strategy)
        path = directory / f"{name}.yaml"
        write_config(
            path,
            k=k,
            tau=tau,
            lambda_w=lambda_w,
            beta_scale=beta_scale,
            strategy=strategy,
        )
        generated.append(path)
    return generated


def generate_baseline_configs() -> list[Path]:
    path = CONFIG_ROOT / "baseline" / f"baseline_{MODEL_NAME}_seed{SEED}.yaml"
    write_baseline_config(path)
    return [path]


def generate_smoke_configs() -> list[Path]:
    return generate_grid(CONFIG_ROOT / "smoke", **SMOKE_GRID)


def generate_full_configs() -> list[Path]:
    return generate_grid(
        CONFIG_ROOT / "full",
        k_values=K_VALUES,
        tau_values=TAU_VALUES,
        lambda_w_values=LAMBDA_W_VALUES,
        beta_scale_values=BETA_SCALE_VALUES,
        training_strategies=TRAINING_STRATEGIES,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate AD-GBC brute-force configs.")
    parser.add_argument(
        "--phase",
        choices=("baseline", "smoke", "full", "all"),
        default="all",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generators = {
        "baseline": generate_baseline_configs,
        "smoke": generate_smoke_configs,
        "full": generate_full_configs,
    }
    selected = ("baseline", "full") if args.phase == "all" else (args.phase,)
    generated: list[Path] = []
    for phase in selected:
        generated.extend(generators[phase]())
    print(f"Generated {len(generated)} AD-GBC configs under {CONFIG_ROOT}")


if __name__ == "__main__":
    main()
