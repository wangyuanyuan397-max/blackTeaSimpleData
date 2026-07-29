"""Generate SupCon + Margin brute-force YAML configs."""

from __future__ import annotations

import argparse
from itertools import product
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
CONFIG_ROOT = THIS_DIR / "configs"

MODEL_NAME = "mixnet_s"
SEED = 2026

# 暴力搜索网格：覆盖 margin、scale、SupCon 温度/权重、LR、projector 维度和分类特征来源。
MARGIN_VALUES = (0.0, 0.05, 0.1, 0.15, 0.2)
SCALE_VALUES = (16.0, 30.0, 64.0)
TEMPERATURE_VALUES = (0.05, 0.1, 0.2)
LAMBDA_SUPCON_VALUES = (0.0, 0.25, 0.5, 1.0)
LR_VALUES = (0.0001, 0.0003, 0.001)
PROJECTOR_OUT_VALUES = (64, 128, 256)
CLASSIFIER_FEATURES = ("projected", "raw")

SMOKE_GRID = {
    "margin_values": (0.0, 0.1),
    "scale_values": (30.0,),
    "temperature_values": (0.1,),
    "lambda_supcon_values": (0.0, 1.0),
    "lr_values": (0.001,),
    "projector_out_values": (128,),
    "classifier_features": ("projected", "raw"),
}


def slug_float(value: float) -> str:
    text = f"{float(value):g}"
    return text.replace("-", "m").replace(".", "p")


def iter_effective_grid(
    *,
    margin_values: tuple[float, ...],
    scale_values: tuple[float, ...],
    temperature_values: tuple[float, ...],
    lambda_supcon_values: tuple[float, ...],
    lr_values: tuple[float, ...],
    projector_out_values: tuple[int, ...],
    classifier_features: tuple[str, ...],
):
    for margin, scale, lambda_supcon, lr, projector_out, classifier_feature in product(
        margin_values,
        scale_values,
        lambda_supcon_values,
        lr_values,
        projector_out_values,
        classifier_features,
    ):
        # lambda_supcon=0 时温度不会参与损失，只保留一个温度值，避免重复实验。
        temperatures = (0.1,) if float(lambda_supcon) == 0.0 else temperature_values
        for temperature in temperatures:
            yield margin, scale, temperature, lambda_supcon, lr, projector_out, classifier_feature


def config_name(
    margin: float,
    scale: float,
    temperature: float,
    lambda_supcon: float,
    lr: float,
    projector_out: int,
    classifier_feature: str,
) -> str:
    return (
        f"supm_{MODEL_NAME}"
        f"_m{slug_float(margin)}"
        f"_s{slug_float(scale)}"
        f"_t{slug_float(temperature)}"
        f"_ls{slug_float(lambda_supcon)}"
        f"_lr{slug_float(lr)}"
        f"_p{int(projector_out)}"
        f"_{classifier_feature}"
    )


def write_supcon_config(
    path: Path,
    *,
    margin: float,
    scale: float,
    temperature: float,
    lambda_supcon: float,
    lr: float,
    projector_out: int,
    classifier_feature: str,
) -> None:
    name = path.stem
    projector_hidden = max(256, int(projector_out) * 2)
    lines = [
        "# SupCon + Margin：训练集使用双视图增强，验证/测试仍走普通单图推理。",
        f"name: {name}",
        f"random_seed: {SEED}",
        "data:",
        "  train_transform:",
        "    # 同一张图独立增强两次，collate 后输入模型为 [B,2,C,H,W]。",
        "    type: two_view_patch_train_224",
        "    image_size: 408",
        "model:",
        "  type: supcon_margin_classifier",
        "  strategy: classification",
        "  backbone:",
        "    type: timm",
        f"    model_name: {MODEL_NAME}",
        "    pretrained: true",
        "    input_size: 408",
        "  projector:",
        f"    hidden_features: {projector_hidden}",
        f"    out_features: {int(projector_out)}",
        "    drop_rate: 0.0",
        f"  classifier_feature: {classifier_feature}",
        "  head:",
        "    type: cosine_margin",
        "    # margin 只在训练 loss 中扣真实类别；推理时使用未扣分的余弦相似度。",
        f"    margin: {float(margin):g}",
        f"    scale: {float(scale):g}",
        "loss:",
        "  type: supcon_margin",
        "  lambda_ce: 1.0",
        "  # lambda_supcon=0 是 margin-only / cosine-CE 控制组。",
        f"  lambda_supcon: {float(lambda_supcon):g}",
        f"  temperature: {float(temperature):g}",
        "optimizer:",
        f"  lr: {float(lr):g}",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def write_baseline_config(path: Path) -> None:
    lines = [
        "# 原始 MixNet-S + Linear + CE baseline；保持单视图训练增强。",
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
        "optimizer:",
        "  lr: 0.001",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def generate_grid(
    directory: Path,
    *,
    margin_values: tuple[float, ...],
    scale_values: tuple[float, ...],
    temperature_values: tuple[float, ...],
    lambda_supcon_values: tuple[float, ...],
    lr_values: tuple[float, ...],
    projector_out_values: tuple[int, ...],
    classifier_features: tuple[str, ...],
) -> list[Path]:
    generated: list[Path] = []
    for margin, scale, temperature, lambda_supcon, lr, projector_out, classifier_feature in iter_effective_grid(
        margin_values=margin_values,
        scale_values=scale_values,
        temperature_values=temperature_values,
        lambda_supcon_values=lambda_supcon_values,
        lr_values=lr_values,
        projector_out_values=projector_out_values,
        classifier_features=classifier_features,
    ):
        name = config_name(
            margin,
            scale,
            temperature,
            lambda_supcon,
            lr,
            projector_out,
            classifier_feature,
        )
        path = directory / f"{name}.yaml"
        write_supcon_config(
            path,
            margin=margin,
            scale=scale,
            temperature=temperature,
            lambda_supcon=lambda_supcon,
            lr=lr,
            projector_out=projector_out,
            classifier_feature=classifier_feature,
        )
        generated.append(path)
    return generated


def generate_baseline_configs() -> list[Path]:
    path = CONFIG_ROOT / "baseline" / f"baseline_{MODEL_NAME}_ce_seed{SEED}.yaml"
    write_baseline_config(path)
    return [path]


def generate_smoke_configs() -> list[Path]:
    return generate_grid(CONFIG_ROOT / "smoke", **SMOKE_GRID)


def generate_full_configs() -> list[Path]:
    return generate_grid(
        CONFIG_ROOT / "full",
        margin_values=MARGIN_VALUES,
        scale_values=SCALE_VALUES,
        temperature_values=TEMPERATURE_VALUES,
        lambda_supcon_values=LAMBDA_SUPCON_VALUES,
        lr_values=LR_VALUES,
        projector_out_values=PROJECTOR_OUT_VALUES,
        classifier_features=CLASSIFIER_FEATURES,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate SupCon + Margin brute-force configs.")
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
    print(f"Generated {len(generated)} SupCon+Margin configs under {CONFIG_ROOT}")


if __name__ == "__main__":
    main()
