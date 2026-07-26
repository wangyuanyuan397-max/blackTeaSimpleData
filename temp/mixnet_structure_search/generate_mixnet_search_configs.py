"""Generate MixNet-S structure search YAML configs."""

from __future__ import annotations

import argparse
from itertools import product
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
CONFIG_ROOT = THIS_DIR / "configs"

ALL_BLOCKS = (
    "S0B0",
    "S1B0", "S1B1",
    "S2B0", "S2B1", "S2B2", "S2B3",
    "S3B0", "S3B1", "S3B2",
    "S4B0", "S4B1", "S4B2",
    "S5B0", "S5B1", "S5B2",
)

STAGE_BLOCKS = {
    0: ("S0B0",),
    1: ("S1B0", "S1B1"),
    2: ("S2B0", "S2B1", "S2B2", "S2B3"),
    3: ("S3B0", "S3B1", "S3B2"),
    4: ("S4B0", "S4B1", "S4B2"),
    5: ("S5B0", "S5B1", "S5B2"),
}

POSITION_EXPERIMENTS = (
    ("p00_original", "ORIGINAL", (3,)),
    ("p01_none_k3", "NONE", (3,)),
    ("p02_all_k357", "ALL", (3, 5, 7)),
    ("p03_stride2_k357", "STRIDE2", (3, 5, 7)),
    ("p04_stride1_k357", "STRIDE1", (3, 5, 7)),
    ("p05_early_s01_k357", "EARLY_S01", (3, 5, 7)),
    ("p06_middle_s23_k357", "MIDDLE_S23", (3, 5, 7)),
    ("p07_last2_s45_k357", "LAST2_S45", (3, 5, 7)),
    ("p08_midlate_s2345_k357", "MIDLATE_S2345", (3, 5, 7)),
    ("p09_late_s345_k357", "LATE_S345", (3, 5, 7)),
    ("p10_only_s0_k357", "ONLY_S0", (3, 5, 7)),
    ("p11_only_s1_k357", "ONLY_S1", (3, 5, 7)),
    ("p12_only_s2_k357", "ONLY_S2", (3, 5, 7)),
    ("p13_only_s3_k357", "ONLY_S3", (3, 5, 7)),
    ("p14_only_s4_k357", "ONLY_S4", (3, 5, 7)),
    ("p15_only_s5_k357", "ONLY_S5", (3, 5, 7)),
    ("p16_first_block_k357", "FIRST_BLOCK", (3, 5, 7)),
    ("p17_repeat_only_k357", "REPEAT_ONLY", (3, 5, 7)),
    ("p18_late_stride2_k357", "LATE_STRIDE2", (3, 5, 7)),
    ("p19_final_downsample_k357", "FINAL_DOWNSAMPLE", (3, 5, 7)),
)

KERNEL_CONTINUOUS = {
    "k35": (3, 5),
    "k357": (3, 5, 7),
    "k3579": (3, 5, 7, 9),
    "k357911": (3, 5, 7, 9, 11),
}

DEFAULT_KERNEL_POSITIONS = (
    ("p03_stride2", "STRIDE2"),
    ("p07_last2_s45", "LAST2_S45"),
    ("p08_midlate_s2345", "MIDLATE_S2345"),
    ("p14_only_s4", "ONLY_S4"),
)

GATE_EXPERIMENTS = (
    ("g0_none", "none"),
    ("g1_static", "static"),
    ("g2_sigmoid", "sigmoid"),
    ("g3_softmax", "softmax"),
)


def kernel_label(kernels: tuple[int, ...]) -> str:
    return "k" + "".join(str(kernel) for kernel in kernels)


def write_config(
    path: Path,
    name: str,
    placement: str,
    kernels: tuple[int, ...],
    gate_type: str = "none",
    kernel_plan: dict[str, tuple[int, ...]] | None = None,
) -> None:
    lines = [
        f"name: {name}",
        "model:",
        "  type: classifier",
        "  strategy: classification",
        "  backbone:",
        "    type: mixnet_s_search",
        "    model_name: mixnet_s",
        "    pretrained: true",
        "    input_size: 408",
        f"    placement: {placement}",
        "    kernel_sizes:",
    ]
    for kernel in kernels:
        lines.append(f"      - {kernel}")
    lines.extend([
        f"    gate_type: {gate_type}",
        "    gate_reduction: 4",
    ])
    if kernel_plan:
        lines.append("    kernel_plan:")
        for block_name in ALL_BLOCKS:
            lines.append(f"      {block_name}:")
            for kernel in kernel_plan[block_name]:
                lines.append(f"        - {kernel}")
    lines.extend([
        "  head:",
        "    type: linear",
        "    drop_rate: 0.0",
        "",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def generate_position_configs() -> list[Path]:
    generated = []
    for index, (slug, placement, kernels) in enumerate(POSITION_EXPERIMENTS):
        path = CONFIG_ROOT / "position" / f"{index:02d}_{slug}.yaml"
        write_config(path, name=path.stem, placement=placement, kernels=kernels)
        generated.append(path)
    return generated


def stage_mask_plan(bits: tuple[int, ...], kernels: tuple[int, ...]) -> dict[str, tuple[int, ...]]:
    selected = set()
    for stage_index, enabled in enumerate(bits):
        if enabled:
            selected.update(STAGE_BLOCKS[stage_index])
    return {
        block_name: kernels if block_name in selected else (3,)
        for block_name in ALL_BLOCKS
    }


def generate_stage_mask_configs() -> list[Path]:
    generated = []
    kernels = (3, 5, 7)
    for bits in product((0, 1), repeat=6):
        mask = "".join(str(bit) for bit in bits)
        path = CONFIG_ROOT / "stage_mask" / f"stagemask_{mask}_{kernel_label(kernels)}.yaml"
        write_config(
            path,
            name=path.stem,
            placement="ORIGINAL",
            kernels=kernels,
            kernel_plan=stage_mask_plan(bits, kernels),
        )
        generated.append(path)
    return generated


def generate_kernel_continuous_configs() -> list[Path]:
    generated = []
    for position_slug, placement in DEFAULT_KERNEL_POSITIONS:
        for kernel_slug, kernels in KERNEL_CONTINUOUS.items():
            path = CONFIG_ROOT / "kernel_continuous" / f"{position_slug}_{kernel_slug}.yaml"
            write_config(path, name=path.stem, placement=placement, kernels=kernels)
            generated.append(path)
    return generated


def generate_gate_configs() -> list[Path]:
    generated = []
    kernels = (3, 5, 7)
    for gate_slug, gate_type in GATE_EXPERIMENTS:
        path = CONFIG_ROOT / "gates" / f"p03_stride2_{kernel_label(kernels)}_{gate_slug}.yaml"
        write_config(
            path,
            name=path.stem,
            placement="STRIDE2",
            kernels=kernels,
            gate_type=gate_type,
        )
        generated.append(path)
    return generated


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate MixNet-S search YAML configs.")
    parser.add_argument(
        "--phase",
        choices=("position", "stage_mask", "kernel_continuous", "gates", "all"),
        default="all",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generators = {
        "position": generate_position_configs,
        "stage_mask": generate_stage_mask_configs,
        "kernel_continuous": generate_kernel_continuous_configs,
        "gates": generate_gate_configs,
    }
    selected = generators.keys() if args.phase == "all" else (args.phase,)
    generated = []
    for phase in selected:
        generated.extend(generators[phase]())
    print(f"Generated {len(generated)} configs under {CONFIG_ROOT}")


if __name__ == "__main__":
    main()
