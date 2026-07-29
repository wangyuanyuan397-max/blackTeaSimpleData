"""Generate follow-up MixNet-S structure-search YAML configs."""

from __future__ import annotations

import argparse
from itertools import product
from pathlib import Path
from typing import Iterable, Mapping, Sequence


THIS_DIR = Path(__file__).resolve().parent
CONFIG_ROOT = THIS_DIR / "configs"

SEEDS = (42, 3407, 2026)
K357 = (3, 5, 7)
KERNEL_OPTIONS: dict[str, tuple[int, ...]] = {
    "k3": (3,),
    "k35": (3, 5),
    "k357": (3, 5, 7),
    "k3579": (3, 5, 7, 9),
}
GATES = (
    ("g0_none", "none"),
    ("g1_static", "static"),
    ("g2_sigmoid", "sigmoid"),
    ("g3_softmax", "softmax"),
)

ALL_BLOCKS = (
    "S0B0",
    "S1B0",
    "S1B1",
    "S2B0",
    "S2B1",
    "S2B2",
    "S2B3",
    "S3B0",
    "S3B1",
    "S3B2",
    "S4B0",
    "S4B1",
    "S4B2",
    "S5B0",
    "S5B1",
    "S5B2",
)
STAGE_BLOCKS = {
    0: ("S0B0",),
    1: ("S1B0", "S1B1"),
    2: ("S2B0", "S2B1", "S2B2", "S2B3"),
    3: ("S3B0", "S3B1", "S3B2"),
    4: ("S4B0", "S4B1", "S4B2"),
    5: ("S5B0", "S5B1", "S5B2"),
}
BLOCK_TO_STAGE = {
    block_name: stage_index
    for stage_index, block_names in STAGE_BLOCKS.items()
    for block_name in block_names
}


def normalize_kernel_sizes(kernel_sizes: Iterable[int]) -> tuple[int, ...]:
    kernels = tuple(int(kernel) for kernel in kernel_sizes)
    if not kernels:
        raise ValueError("kernel_sizes cannot be empty.")
    if len(set(kernels)) != len(kernels):
        raise ValueError(f"Duplicate kernel sizes are not allowed: {kernels}")
    if any(kernel < 1 or kernel % 2 != 1 for kernel in kernels):
        raise ValueError(f"Kernel sizes must be positive odd integers: {kernels}")
    return kernels


def stage_kernel_plan(
    stage_kernels: Mapping[int, Sequence[int]],
    default_kernels: Sequence[int] = (3,),
) -> dict[str, tuple[int, ...]]:
    normalized_default = normalize_kernel_sizes(default_kernels)
    normalized_stage_kernels = {
        int(stage_index): normalize_kernel_sizes(kernels)
        for stage_index, kernels in stage_kernels.items()
    }
    return {
        block_name: normalized_stage_kernels.get(
            BLOCK_TO_STAGE[block_name],
            normalized_default,
        )
        for block_name in ALL_BLOCKS
    }


def write_config(
    path: Path,
    name: str,
    *,
    placement: str,
    kernel_sizes: Sequence[int],
    gate_type: str = "none",
    random_seed: int | None = None,
    kernel_plan: Mapping[str, Sequence[int]] | None = None,
) -> None:
    lines: list[str] = [f"name: {name}"]
    if random_seed is not None:
        lines.append(f"random_seed: {int(random_seed)}")
    lines.extend(
        [
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
    )
    for kernel in normalize_kernel_sizes(kernel_sizes):
        lines.append(f"      - {kernel}")
    lines.extend(
        [
            f"    gate_type: {gate_type}",
            "    gate_reduction: 4",
        ]
    )
    if kernel_plan:
        lines.append("    kernel_plan:")
        for block_name in ALL_BLOCKS:
            lines.append(f"      {block_name}:")
            for kernel in normalize_kernel_sizes(kernel_plan[block_name]):
                lines.append(f"        - {kernel}")
    lines.extend(
        [
            "  head:",
            "    type: linear",
            "    drop_rate: 0.0",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def only_s0_kwargs(gate_type: str = "none") -> dict[str, object]:
    return {
        "placement": "ONLY_S0",
        "kernel_sizes": K357,
        "gate_type": gate_type,
    }


def s235_kwargs(gate_type: str = "none") -> dict[str, object]:
    return {
        "placement": "ORIGINAL",
        "kernel_sizes": K357,
        "gate_type": gate_type,
        "kernel_plan": stage_kernel_plan({2: K357, 3: K357, 5: K357}),
    }


def original_kwargs() -> dict[str, object]:
    return {
        "placement": "ORIGINAL",
        "kernel_sizes": (3,),
        "gate_type": "none",
    }


def stride2_softmax_kwargs() -> dict[str, object]:
    return {
        "placement": "STRIDE2",
        "kernel_sizes": K357,
        "gate_type": "softmax",
    }


def generate_seed_repro_configs() -> list[Path]:
    output_dir = CONFIG_ROOT / "followup_seed_repro"
    structures = (
        ("repro_original_mixnet_s", original_kwargs()),
        ("repro_only_s0_k357", only_s0_kwargs()),
        ("repro_s235_k357", s235_kwargs()),
        ("repro_stride2_k357_g3_softmax", stride2_softmax_kwargs()),
    )
    generated: list[Path] = []
    for seed in SEEDS:
        for slug, kwargs in structures:
            name = f"{slug}_seed{seed}"
            path = output_dir / f"{name}.yaml"
            write_config(path, name, random_seed=seed, **kwargs)
            generated.append(path)
    return generated


def generate_champion_gate_configs() -> list[Path]:
    output_dir = CONFIG_ROOT / "followup_champion_gates"
    generated: list[Path] = []
    for structure_slug, factory in (
        ("only_s0_k357", only_s0_kwargs),
        ("s235_k357", s235_kwargs),
    ):
        for gate_slug, gate_type in GATES:
            name = f"gate_{structure_slug}_{gate_slug}"
            path = output_dir / f"{name}.yaml"
            write_config(path, name, random_seed=2026, **factory(gate_type))
            generated.append(path)
    return generated


def generate_s235_kernel_grid_configs() -> list[Path]:
    output_dir = CONFIG_ROOT / "followup_s235_kernel_grid"
    generated: list[Path] = []
    for s2_key, s3_key, s5_key in product(KERNEL_OPTIONS, repeat=3):
        name = f"s235grid_s2{s2_key}_s3{s3_key}_s5{s5_key}"
        path = output_dir / f"{name}.yaml"
        kernel_plan = stage_kernel_plan(
            {
                2: KERNEL_OPTIONS[s2_key],
                3: KERNEL_OPTIONS[s3_key],
                5: KERNEL_OPTIONS[s5_key],
            }
        )
        write_config(
            path,
            name,
            random_seed=2026,
            placement="ORIGINAL",
            kernel_sizes=K357,
            gate_type="none",
            kernel_plan=kernel_plan,
        )
        generated.append(path)
    return generated


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate next-priority MixNet-S follow-up configs."
    )
    parser.add_argument(
        "--phase",
        choices=("seed_repro", "champion_gates", "s235_kernel_grid", "all"),
        default="all",
        help="Follow-up config group to generate.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generators = {
        "seed_repro": generate_seed_repro_configs,
        "champion_gates": generate_champion_gate_configs,
        "s235_kernel_grid": generate_s235_kernel_grid_configs,
    }
    selected = generators.keys() if args.phase == "all" else (args.phase,)
    generated: list[Path] = []
    for phase in selected:
        generated.extend(generators[phase]())
    print(f"Generated {len(generated)} follow-up configs under {CONFIG_ROOT}")


if __name__ == "__main__":
    main()
