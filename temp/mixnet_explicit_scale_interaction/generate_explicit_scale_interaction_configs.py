"""Generate MixNet-S explicit scale-interaction brute-force YAML configs."""

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

REQUESTED_OPERATORS = (
    "baseline",
    "scale_attention",
    "small_to_large_guidance",
    "large_to_small_guidance",
    "weighted_sum",
    "cross_residual_bidir",
)

EXTRA_OPERATORS = (
    "full_concat_interaction",
    "full_weighted_interaction",
)


def yaml_scalar(value) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    return str(value)


def stage_mask_label(stage_mask: Sequence[bool]) -> str:
    return "".join("1" if value else "0" for value in stage_mask)


def make_experiment_name(
    interaction_type: str,
    placement: str = "ALL",
    edge_mode: str = "adjacent",
    residual_strength: float = 1.0,
    attention_hidden_dim: int = 8,
    stage_mask: Sequence[bool] | None = None,
    block_indices: Sequence[int | str] | None = None,
) -> str:
    parts = [interaction_type]
    if interaction_type != "baseline":
        if placement != "ALL":
            parts.append(f"place-{placement.lower()}")
        if edge_mode != "adjacent":
            parts.append(f"edge-{edge_mode}")
        if abs(float(residual_strength) - 1.0) > 1e-12:
            parts.append(f"rs{float(residual_strength):.2f}".replace(".", "p"))
        if int(attention_hidden_dim) != 8:
            parts.append(f"ah{int(attention_hidden_dim)}")
    if stage_mask is not None:
        parts.append(f"stage-{stage_mask_label(stage_mask)}")
    if block_indices is not None:
        block_label = "-".join(str(item).lower() for item in block_indices)
        parts.append(f"blocks-{block_label}")
    return "__".join(parts)


def write_config(
    path: Path,
    name: str,
    interaction_type: str,
    placement: str = "ALL",
    edge_mode: str = "adjacent",
    residual_strength: float = 1.0,
    attention_hidden_dim: int = 8,
    stage_mask: Sequence[bool] | None = None,
    block_indices: Sequence[int | str] | None = None,
) -> None:
    lines = [
        f"name: {name}",
        "model:",
        "  type: classifier",
        "  strategy: classification",
        "  backbone:",
        "    type: mixnet_s_explicit_scale_interaction",
        "    model_name: mixnet_s",
        "    pretrained: true",
        "    input_size: 408",
        f"    interaction_type: {interaction_type}",
        f"    placement: {placement}",
        f"    edge_mode: {edge_mode}",
        f"    residual_strength: {float(residual_strength):.4f}",
        f"    attention_hidden_dim: {int(attention_hidden_dim)}",
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


def generate_operator_configs(include_extra: bool = True) -> list[Path]:
    operators = list(REQUESTED_OPERATORS)
    if include_extra:
        operators.extend(EXTRA_OPERATORS)

    generated = []
    for index, interaction_type in enumerate(operators):
        name = make_experiment_name(interaction_type)
        path = CONFIG_ROOT / "operators" / f"{index:02d}_{name}.yaml"
        write_config(
            path,
            name=path.stem,
            interaction_type=interaction_type,
        )
        generated.append(path)
    return generated


def generate_edge_mode_configs() -> list[Path]:
    specs = []
    for interaction_type in (
        "small_to_large_guidance",
        "large_to_small_guidance",
        "cross_residual_bidir",
        "full_concat_interaction",
        "full_weighted_interaction",
    ):
        for edge_mode in ("adjacent", "all"):
            specs.append((interaction_type, edge_mode))

    generated = []
    for index, (interaction_type, edge_mode) in enumerate(specs):
        name = make_experiment_name(interaction_type, edge_mode=edge_mode)
        path = CONFIG_ROOT / "edge_modes" / f"edge_{index:02d}_{name}.yaml"
        write_config(
            path,
            name=path.stem,
            interaction_type=interaction_type,
            edge_mode=edge_mode,
        )
        generated.append(path)
    return generated


def generate_stage_single_configs(
    interaction_type: str,
    edge_mode: str,
    residual_strength: float,
    attention_hidden_dim: int,
) -> list[Path]:
    generated = []
    for stage_index in range(STAGE_COUNT):
        mask = tuple(index == stage_index for index in range(STAGE_COUNT))
        name = make_experiment_name(
            interaction_type,
            edge_mode=edge_mode,
            residual_strength=residual_strength,
            attention_hidden_dim=attention_hidden_dim,
            stage_mask=mask,
        )
        path = CONFIG_ROOT / "stage_single" / f"{stage_index:02d}_{name}.yaml"
        write_config(
            path,
            name=path.stem,
            interaction_type=interaction_type,
            edge_mode=edge_mode,
            residual_strength=residual_strength,
            attention_hidden_dim=attention_hidden_dim,
            stage_mask=mask,
        )
        generated.append(path)

    all_mask = tuple(True for _ in range(STAGE_COUNT))
    name = make_experiment_name(
        interaction_type,
        edge_mode=edge_mode,
        residual_strength=residual_strength,
        attention_hidden_dim=attention_hidden_dim,
        stage_mask=all_mask,
    )
    path = CONFIG_ROOT / "stage_single" / f"{STAGE_COUNT:02d}_{name}.yaml"
    write_config(
        path,
        name=path.stem,
        interaction_type=interaction_type,
        edge_mode=edge_mode,
        residual_strength=residual_strength,
        attention_hidden_dim=attention_hidden_dim,
        stage_mask=all_mask,
    )
    generated.append(path)
    return generated


def generate_stage_mask_configs(
    interaction_type: str,
    edge_mode: str,
    residual_strength: float,
    attention_hidden_dim: int,
) -> list[Path]:
    generated = []
    for bits in product((False, True), repeat=STAGE_COUNT):
        if not any(bits):
            continue
        name = make_experiment_name(
            interaction_type,
            edge_mode=edge_mode,
            residual_strength=residual_strength,
            attention_hidden_dim=attention_hidden_dim,
            stage_mask=bits,
        )
        path = CONFIG_ROOT / "stage_mask" / f"stagemask_{stage_mask_label(bits)}__{name}.yaml"
        write_config(
            path,
            name=path.stem,
            interaction_type=interaction_type,
            edge_mode=edge_mode,
            residual_strength=residual_strength,
            attention_hidden_dim=attention_hidden_dim,
            stage_mask=bits,
        )
        generated.append(path)
    return generated


def generate_block_single_configs(
    interaction_type: str,
    edge_mode: str,
    residual_strength: float,
    attention_hidden_dim: int,
) -> list[Path]:
    generated = []
    for index, block_name in enumerate(ALL_BLOCKS):
        name = make_experiment_name(
            interaction_type,
            edge_mode=edge_mode,
            residual_strength=residual_strength,
            attention_hidden_dim=attention_hidden_dim,
            block_indices=(block_name,),
        )
        path = CONFIG_ROOT / "block_single" / f"{index:02d}_{name}.yaml"
        write_config(
            path,
            name=path.stem,
            interaction_type=interaction_type,
            edge_mode=edge_mode,
            residual_strength=residual_strength,
            attention_hidden_dim=attention_hidden_dim,
            block_indices=(block_name,),
        )
        generated.append(path)
    return generated


def generate_block_subset_configs(
    interaction_type: str,
    edge_mode: str,
    residual_strength: float,
    attention_hidden_dim: int,
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
                interaction_type,
                edge_mode=edge_mode,
                residual_strength=residual_strength,
                attention_hidden_dim=attention_hidden_dim,
                block_indices=subset,
            )
            path = CONFIG_ROOT / "block_subset" / f"{counter:03d}_{name}.yaml"
            write_config(
                path,
                name=path.stem,
                interaction_type=interaction_type,
                edge_mode=edge_mode,
                residual_strength=residual_strength,
                attention_hidden_dim=attention_hidden_dim,
                block_indices=subset,
            )
            generated.append(path)
            counter += 1
    return generated


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate explicit scale-interaction YAML configs.")
    parser.add_argument(
        "--phase",
        choices=("operators", "edge_modes", "stage_single", "stage_mask", "block_single", "block_subset", "all"),
        default="all",
    )
    parser.add_argument("--interaction-type", default="cross_residual_bidir")
    parser.add_argument("--edge-mode", choices=("adjacent", "all"), default="adjacent")
    parser.add_argument("--residual-strength", type=float, default=1.0)
    parser.add_argument("--attention-hidden-dim", type=int, default=8)
    parser.add_argument("--subset-blocks", nargs="*", default=())
    parser.add_argument("--max-subset-size", type=int, default=None)
    parser.add_argument("--no-extra-operators", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generated: list[Path] = []
    phases: Iterable[str]
    if args.phase == "all":
        phases = ("operators", "edge_modes")
    else:
        phases = (args.phase,)

    for phase in phases:
        if phase == "operators":
            generated.extend(generate_operator_configs(include_extra=not args.no_extra_operators))
        elif phase == "edge_modes":
            generated.extend(generate_edge_mode_configs())
        elif phase == "stage_single":
            generated.extend(
                generate_stage_single_configs(
                    interaction_type=args.interaction_type,
                    edge_mode=args.edge_mode,
                    residual_strength=args.residual_strength,
                    attention_hidden_dim=args.attention_hidden_dim,
                )
            )
        elif phase == "stage_mask":
            generated.extend(
                generate_stage_mask_configs(
                    interaction_type=args.interaction_type,
                    edge_mode=args.edge_mode,
                    residual_strength=args.residual_strength,
                    attention_hidden_dim=args.attention_hidden_dim,
                )
            )
        elif phase == "block_single":
            generated.extend(
                generate_block_single_configs(
                    interaction_type=args.interaction_type,
                    edge_mode=args.edge_mode,
                    residual_strength=args.residual_strength,
                    attention_hidden_dim=args.attention_hidden_dim,
                )
            )
        elif phase == "block_subset":
            generated.extend(
                generate_block_subset_configs(
                    interaction_type=args.interaction_type,
                    edge_mode=args.edge_mode,
                    residual_strength=args.residual_strength,
                    attention_hidden_dim=args.attention_hidden_dim,
                    subset_blocks=args.subset_blocks,
                    max_subset_size=args.max_subset_size,
                )
            )
        else:
            raise AssertionError(f"Unhandled phase: {phase}")

    print(f"Generated {len(generated)} configs under {CONFIG_ROOT}")


if __name__ == "__main__":
    main()
