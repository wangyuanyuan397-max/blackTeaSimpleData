"""Generate MixNet-S channel-shuffle brute-force search YAML configs."""

from __future__ import annotations

import argparse
from itertools import combinations, product
from pathlib import Path
from typing import Iterable, Sequence


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

STAGE_COUNT = 6


def yaml_scalar(value) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    return str(value)


def stage_mask_label(stage_mask: Sequence[bool]) -> str:
    return "".join("1" if value else "0" for value in stage_mask)


def make_experiment_name(
    fusion_type: str,
    target_groups: int = 1,
    insertion_mode: str = "replace",
    partial_ratio: float | None = None,
    stage_mask: Sequence[bool] | None = None,
    block_indices: Sequence[int | str] | None = None,
    shuffle_mode: str = "scale",
) -> str:
    parts = [fusion_type]
    if fusion_type != "baseline":
        if fusion_type == "partial_mix":
            parts.append(f"r{float(partial_ratio or 0.5):.2f}".replace(".", "p"))
        elif target_groups > 1:
            parts.append(f"g{int(target_groups)}")
        parts.append(f"mode-{insertion_mode}")
        if shuffle_mode != "scale":
            parts.append(f"shuffle-{shuffle_mode}")
    if stage_mask is not None:
        parts.append(f"stage-{stage_mask_label(stage_mask)}")
    if block_indices is not None:
        block_label = "-".join(str(item).lower() for item in block_indices)
        parts.append(f"blocks-{block_label}")
    return "__".join(parts)


def write_config(
    path: Path,
    name: str,
    fusion_type: str,
    target_groups: int = 1,
    partial_ratio: float = 0.5,
    placement: str = "ALL",
    stage_mask: Sequence[bool] | None = None,
    block_indices: Sequence[int | str] | None = None,
    shuffle_mode: str = "scale",
    insertion_mode: str = "replace",
    random_permutation_seed: int = 2026,
) -> None:
    lines = [
        f"name: {name}",
        "model:",
        "  type: classifier",
        "  strategy: classification",
        "  backbone:",
        "    type: mixnet_s_channel_shuffle_search",
        "    model_name: mixnet_s",
        "    pretrained: true",
        "    input_size: 408",
        f"    fusion_type: {fusion_type}",
        f"    target_groups: {int(target_groups)}",
        f"    partial_ratio: {float(partial_ratio):.4f}",
        f"    placement: {placement}",
        f"    shuffle_mode: {shuffle_mode}",
        f"    insertion_mode: {insertion_mode}",
        f"    random_permutation_seed: {int(random_permutation_seed)}",
    ]
    if stage_mask is not None:
        lines.append("    stage_mask:")
        for value in stage_mask:
            lines.append(f"      - {yaml_scalar(bool(value))}")
    if block_indices is not None:
        lines.append("    block_indices:")
        for value in block_indices:
            if isinstance(value, str):
                lines.append(f"      - {value}")
            else:
                lines.append(f"      - {int(value)}")
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


def generate_operator_configs(shuffle_mode: str = "scale") -> list[Path]:
    specs = [
        ("baseline", 1, "replace", None),
        ("group", 2, "replace", None),
        ("group", 4, "replace", None),
        ("group", 8, "replace", None),
        ("shuffle_group", 2, "replace", None),
        ("shuffle_group", 4, "replace", None),
        ("shuffle_group", 8, "replace", None),
        ("group_shuffle", 2, "replace", None),
        ("group_shuffle", 4, "replace", None),
        ("group_shuffle", 8, "replace", None),
        ("shuffle_dense", 1, "replace", None),
    ]
    generated = []
    for index, (fusion_type, groups, insertion_mode, ratio) in enumerate(specs):
        name = make_experiment_name(
            fusion_type,
            target_groups=groups,
            insertion_mode=insertion_mode,
            partial_ratio=ratio,
            shuffle_mode=shuffle_mode,
        )
        path = CONFIG_ROOT / "operators" / f"{index:02d}_{name}.yaml"
        write_config(
            path,
            name=path.stem,
            fusion_type=fusion_type,
            target_groups=groups,
            insertion_mode=insertion_mode,
            partial_ratio=ratio or 0.5,
            shuffle_mode=shuffle_mode,
        )
        generated.append(path)
    return generated


def generate_additive_configs(shuffle_mode: str = "scale") -> list[Path]:
    specs = []
    for groups in (2, 4, 8):
        specs.append(("extra_group", groups, "add", None))
        specs.append(("extra_shuffle_group", groups, "add", None))
    for ratio in (0.25, 0.50, 0.75):
        specs.append(("partial_mix", 1, "add", ratio))

    generated = []
    for index, (fusion_type, groups, insertion_mode, ratio) in enumerate(specs):
        name = make_experiment_name(
            fusion_type,
            target_groups=groups,
            insertion_mode=insertion_mode,
            partial_ratio=ratio,
            shuffle_mode=shuffle_mode,
        )
        path = CONFIG_ROOT / "additive" / f"{index:02d}_{name}.yaml"
        write_config(
            path,
            name=path.stem,
            fusion_type=fusion_type,
            target_groups=groups,
            partial_ratio=ratio or 0.5,
            insertion_mode=insertion_mode,
            shuffle_mode=shuffle_mode,
        )
        generated.append(path)
    return generated


def generate_stage_single_configs(
    fusion_type: str,
    target_groups: int,
    partial_ratio: float,
    shuffle_mode: str,
) -> list[Path]:
    generated = []
    for stage_index in range(STAGE_COUNT):
        mask = tuple(index == stage_index for index in range(STAGE_COUNT))
        name = make_experiment_name(
            fusion_type,
            target_groups=target_groups,
            insertion_mode=insertion_mode_for(fusion_type),
            partial_ratio=partial_ratio,
            stage_mask=mask,
            shuffle_mode=shuffle_mode,
        )
        path = CONFIG_ROOT / "stage_single" / f"{stage_index:02d}_{name}.yaml"
        write_config(
            path,
            name=path.stem,
            fusion_type=fusion_type,
            target_groups=target_groups,
            partial_ratio=partial_ratio,
            stage_mask=mask,
            insertion_mode=insertion_mode_for(fusion_type),
            shuffle_mode=shuffle_mode,
        )
        generated.append(path)
    all_mask = tuple(True for _ in range(STAGE_COUNT))
    name = make_experiment_name(
        fusion_type,
        target_groups=target_groups,
        insertion_mode=insertion_mode_for(fusion_type),
        partial_ratio=partial_ratio,
        stage_mask=all_mask,
        shuffle_mode=shuffle_mode,
    )
    path = CONFIG_ROOT / "stage_single" / f"{STAGE_COUNT:02d}_{name}.yaml"
    write_config(
        path,
        name=path.stem,
        fusion_type=fusion_type,
        target_groups=target_groups,
        partial_ratio=partial_ratio,
        stage_mask=all_mask,
        insertion_mode=insertion_mode_for(fusion_type),
        shuffle_mode=shuffle_mode,
    )
    generated.append(path)
    return generated


def generate_stage_mask_configs(
    fusion_type: str,
    target_groups: int,
    partial_ratio: float,
    shuffle_mode: str,
) -> list[Path]:
    generated = []
    for bits in product((False, True), repeat=STAGE_COUNT):
        if not any(bits):
            continue
        name = make_experiment_name(
            fusion_type,
            target_groups=target_groups,
            insertion_mode=insertion_mode_for(fusion_type),
            partial_ratio=partial_ratio,
            stage_mask=bits,
            shuffle_mode=shuffle_mode,
        )
        path = CONFIG_ROOT / "stage_mask" / f"stagemask_{stage_mask_label(bits)}__{name}.yaml"
        write_config(
            path,
            name=path.stem,
            fusion_type=fusion_type,
            target_groups=target_groups,
            partial_ratio=partial_ratio,
            stage_mask=bits,
            insertion_mode=insertion_mode_for(fusion_type),
            shuffle_mode=shuffle_mode,
        )
        generated.append(path)
    return generated


def generate_block_single_configs(
    fusion_type: str,
    target_groups: int,
    partial_ratio: float,
    shuffle_mode: str,
) -> list[Path]:
    generated = []
    for index, block_name in enumerate(ALL_BLOCKS):
        name = make_experiment_name(
            fusion_type,
            target_groups=target_groups,
            insertion_mode=insertion_mode_for(fusion_type),
            partial_ratio=partial_ratio,
            block_indices=(block_name,),
            shuffle_mode=shuffle_mode,
        )
        path = CONFIG_ROOT / "block_single" / f"{index:02d}_{name}.yaml"
        write_config(
            path,
            name=path.stem,
            fusion_type=fusion_type,
            target_groups=target_groups,
            partial_ratio=partial_ratio,
            block_indices=(block_name,),
            insertion_mode=insertion_mode_for(fusion_type),
            shuffle_mode=shuffle_mode,
        )
        generated.append(path)
    return generated


def generate_block_subset_configs(
    fusion_type: str,
    target_groups: int,
    partial_ratio: float,
    shuffle_mode: str,
    subset_blocks: Sequence[str],
    max_subset_size: int | None,
) -> list[Path]:
    if not subset_blocks:
        raise ValueError("--subset-blocks is required for phase block_subset.")
    blocks = tuple(str(block).upper() for block in subset_blocks)
    unknown = [block for block in blocks if block not in ALL_BLOCKS]
    if unknown:
        raise ValueError(f"Unknown MixNet-S block(s): {unknown}")

    generated = []
    counter = 0
    max_size = len(blocks) if max_subset_size is None else min(len(blocks), int(max_subset_size))
    for size in range(1, max_size + 1):
        for subset in combinations(blocks, size):
            name = make_experiment_name(
                fusion_type,
                target_groups=target_groups,
                insertion_mode=insertion_mode_for(fusion_type),
                partial_ratio=partial_ratio,
                block_indices=subset,
                shuffle_mode=shuffle_mode,
            )
            path = CONFIG_ROOT / "block_subset" / f"{counter:03d}_{name}.yaml"
            write_config(
                path,
                name=path.stem,
                fusion_type=fusion_type,
                target_groups=target_groups,
                partial_ratio=partial_ratio,
                block_indices=subset,
                insertion_mode=insertion_mode_for(fusion_type),
                shuffle_mode=shuffle_mode,
            )
            generated.append(path)
            counter += 1
    return generated


def insertion_mode_for(fusion_type: str) -> str:
    return "add" if str(fusion_type).lower() in {"extra_group", "extra_shuffle_group", "partial_mix"} else "replace"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate channel-shuffle brute-force YAML configs.")
    parser.add_argument(
        "--phase",
        choices=("operators", "additive", "stage_single", "stage_mask", "block_single", "block_subset", "all"),
        default="all",
    )
    parser.add_argument("--fusion-type", default="shuffle_group")
    parser.add_argument("--target-groups", type=int, default=4)
    parser.add_argument("--partial-ratio", type=float, default=0.5)
    parser.add_argument(
        "--shuffle-mode",
        choices=("none", "scale", "fusion", "random", "double_scale"),
        default="scale",
    )
    parser.add_argument("--subset-blocks", nargs="*", default=())
    parser.add_argument("--max-subset-size", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generated: list[Path] = []
    phases: Iterable[str]
    if args.phase == "all":
        phases = ("operators", "additive")
    else:
        phases = (args.phase,)

    for phase in phases:
        if phase == "operators":
            generated.extend(generate_operator_configs(shuffle_mode=args.shuffle_mode))
        elif phase == "additive":
            generated.extend(generate_additive_configs(shuffle_mode=args.shuffle_mode))
        elif phase == "stage_single":
            generated.extend(
                generate_stage_single_configs(
                    fusion_type=args.fusion_type,
                    target_groups=args.target_groups,
                    partial_ratio=args.partial_ratio,
                    shuffle_mode=args.shuffle_mode,
                )
            )
        elif phase == "stage_mask":
            generated.extend(
                generate_stage_mask_configs(
                    fusion_type=args.fusion_type,
                    target_groups=args.target_groups,
                    partial_ratio=args.partial_ratio,
                    shuffle_mode=args.shuffle_mode,
                )
            )
        elif phase == "block_single":
            generated.extend(
                generate_block_single_configs(
                    fusion_type=args.fusion_type,
                    target_groups=args.target_groups,
                    partial_ratio=args.partial_ratio,
                    shuffle_mode=args.shuffle_mode,
                )
            )
        elif phase == "block_subset":
            generated.extend(
                generate_block_subset_configs(
                    fusion_type=args.fusion_type,
                    target_groups=args.target_groups,
                    partial_ratio=args.partial_ratio,
                    shuffle_mode=args.shuffle_mode,
                    subset_blocks=args.subset_blocks,
                    max_subset_size=args.max_subset_size,
                )
            )
        else:
            raise AssertionError(f"Unhandled phase: {phase}")

    print(f"Generated {len(generated)} configs under {CONFIG_ROOT}")


if __name__ == "__main__":
    main()
