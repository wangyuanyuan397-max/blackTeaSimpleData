"""Generate 32 x 3 MixNet-S deformable-attention sweep YAML configs."""

from __future__ import annotations

from itertools import product
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
CONFIG_ROOT = THIS_DIR / "configs"
SEEDS = (42, 3407, 2026)
STAGE_COUNT = 5
DEFORM_PARAMS = {
    "num_heads": 4,
    "num_points": 4,
    "max_offset": 2.0,
    "attention_dropout": 0.0,
    "projection_dropout": 0.0,
    "layer_scale_init": 0.001,
}


def stage_ids_from_bits(bits: str) -> list[int]:
    if len(bits) != STAGE_COUNT or any(char not in "01" for char in bits):
        raise ValueError(f"Expected a {STAGE_COUNT}-bit string, got {bits!r}.")
    return [index for index, char in enumerate(bits) if char == "1"]


def write_config(path: Path, bits: str, seed: int) -> None:
    stage_ids = stage_ids_from_bits(bits)
    run_name = path.stem
    is_baseline = not stage_ids
    backbone_type = "timm" if is_baseline else "mixnet_s_deformable"
    lines = [
        f"name: {run_name}",
        f"random_seed: {seed}",
        "model:",
        "  type: classifier",
        "  strategy: classification",
        "  backbone:",
        f"    type: {backbone_type}",
        "    model_name: mixnet_s",
        "    pretrained: true",
        "    input_size: 408",
    ]
    if not is_baseline:
        lines.append("    deform_stage_ids:")
        lines.extend(f"      - {stage_id}" for stage_id in stage_ids)
        for key, value in DEFORM_PARAMS.items():
            lines.append(f"    {key}: {value}")
        lines.append("    deform_checkpoint: false")
    lines.extend(
        [
            "  head:",
            "    type: linear",
            "    drop_rate: 0.0",
            "deformable_attention:",
            f"  stage_bits: \"{bits}\"",
            "  bit_order: [S0, S1, S2, S3, S4]",
        ]
    )
    if stage_ids:
        lines.append("  deform_stage_ids:")
        lines.extend(f"    - {stage_id}" for stage_id in stage_ids)
    else:
        lines.append("  deform_stage_ids: []")
    for key, value in DEFORM_PARAMS.items():
        lines.append(f"  {key}: {value}")
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def generate_configs() -> list[Path]:
    generated: list[Path] = []
    for bits_tuple in product("01", repeat=STAGE_COUNT):
        bits = "".join(bits_tuple)
        for seed in SEEDS:
            path = CONFIG_ROOT / f"D{bits}_seed{seed}.yaml"
            write_config(path, bits, seed)
            generated.append(path)
    return generated


def main() -> None:
    generated = generate_configs()
    print(f"Generated {len(generated)} configs under {CONFIG_ROOT}")


if __name__ == "__main__":
    main()
