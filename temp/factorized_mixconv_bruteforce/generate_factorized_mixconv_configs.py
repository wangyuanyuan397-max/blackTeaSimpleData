"""Generate the 8 x 3 Factorized MixConv brute-force YAML configs."""

from __future__ import annotations

from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
CONFIG_ROOT = THIS_DIR / "configs"

SEARCH_CONFIGS: dict[str, tuple[int, ...]] = {
    "F000": (),
    "F001": (9,),
    "F010": (7,),
    "F011": (7, 9),
    "F100": (5,),
    "F101": (5, 9),
    "F110": (5, 7),
    "F111": (5, 7, 9),
}

SEEDS = (42, 3407, 2026)


def write_config(path: Path, config_name: str, kernels: tuple[int, ...], seed: int) -> None:
    run_name = path.stem
    lines = [
        f"name: {run_name}",
        f"random_seed: {seed}",
        "model:",
        "  type: classifier",
        "  strategy: classification",
        "  backbone:",
        "    type: mixnet_s_factorized",
        "    model_name: mixnet_s",
        "    pretrained: true",
        "    input_size: 408",
    ]
    if kernels:
        lines.append("    factorized_kernels:")
        lines.extend(f"      - {kernel}" for kernel in kernels)
    else:
        lines.append("    factorized_kernels: []")
    lines.extend(
        [
            "  head:",
            "    type: linear",
            "    drop_rate: 0.0",
            "factorized_mixconv:",
            f"  config_name: {config_name}",
            "  bit_order: [5, 7, 9]",
        ]
    )
    if kernels:
        lines.append("  factorized_kernels:")
        lines.extend(f"    - {kernel}" for kernel in kernels)
    else:
        lines.append("  factorized_kernels: []")
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def generate_configs() -> list[Path]:
    generated: list[Path] = []
    for config_name, kernels in SEARCH_CONFIGS.items():
        for seed in SEEDS:
            path = CONFIG_ROOT / f"{config_name}_seed{seed}.yaml"
            write_config(path, config_name, kernels, seed)
            generated.append(path)
    return generated


def main() -> None:
    generated = generate_configs()
    print(f"Generated {len(generated)} configs under {CONFIG_ROOT}")


if __name__ == "__main__":
    main()
