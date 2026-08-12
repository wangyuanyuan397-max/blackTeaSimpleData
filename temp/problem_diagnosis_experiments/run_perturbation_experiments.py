"""用同一个 checkpoint 做遮挡、布局、颜色和纹理受控破坏实验。"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path


EXPERIMENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EXPERIMENT_DIR.parents[1]
sys.path.insert(0, str(EXPERIMENT_DIR))

from diagnosis_common import (  # noqa: E402
    DEFAULT_CLASSES,
    DiagnosticImageDataset,
    DiagnosticTransform,
    PerturbationSpec,
    evaluate_model,
    load_checkpoint,
    make_loader,
    metrics_row,
    parse_float_list,
    resolve_device,
    set_random_seed,
    write_csv,
    write_json,
)


def build_conditions(args) -> list[PerturbationSpec]:
    """创建诊断条件；随机条件用多个重复估计均值和波动。"""

    conditions = [PerturbationSpec("original")]
    conditions.append(PerturbationSpec("grayscale"))
    conditions.extend(PerturbationSpec("blur", radius) for radius in parse_float_list(args.blur_radii))
    for repeat in range(args.repeats):
        conditions.extend(
            PerturbationSpec("occlusion", ratio, repeat=repeat)
            for ratio in parse_float_list(args.occlusion_ratios)
        )
        conditions.append(PerturbationSpec("color_jitter", args.color_jitter, repeat=repeat))
        conditions.append(PerturbationSpec("patch_shuffle", grid=args.patch_grid, repeat=repeat))
        conditions.append(
            PerturbationSpec(
                "texture_shuffle",
                grid=args.texture_macro_grid,
                micro_grid=args.texture_micro_grid,
                repeat=repeat,
            )
        )
    return conditions


def condition_name(spec: PerturbationSpec) -> str:
    """生成不含歧义的输出目录名。"""

    if spec.name == "original":
        return "original"
    if spec.name in {"occlusion", "blur", "color_jitter"}:
        value = str(spec.value).replace(".", "p")
        base = f"{spec.name}_{value}"
    elif spec.name == "patch_shuffle":
        base = f"patch_shuffle_{spec.grid}x{spec.grid}"
    elif spec.name == "texture_shuffle":
        base = f"texture_shuffle_macro{spec.grid}_micro{spec.micro_grid}"
    else:
        base = spec.name
    randomized = spec.name in {"occlusion", "color_jitter", "patch_shuffle", "texture_shuffle"}
    return f"{base}_repeat{spec.repeat + 1}" if randomized else base


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="checkpoint 受控破坏诊断实验。")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model-name", default=None, help="旧 checkpoint 无元数据时填写。")
    parser.add_argument("--num-classes", type=int, default=None)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=PROJECT_ROOT / "datasets_01234_original_split",
    )
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--crop-size", type=int, default=None, help="默认读取 checkpoint 元数据。")
    parser.add_argument("--input-size", type=int, default=None, help="默认读取 checkpoint 元数据。")
    parser.add_argument(
        "--full-image",
        action="store_true",
        help="按整张原图评估；Global checkpoint 会从元数据自动识别。",
    )
    parser.add_argument("--views", type=int, default=9)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--occlusion-ratios", default="0.1,0.2,0.3,0.4,0.5")
    parser.add_argument("--blur-radii", default="1.5,3.0")
    parser.add_argument("--color-jitter", type=float, default=0.3)
    parser.add_argument("--patch-grid", type=int, default=3)
    parser.add_argument("--texture-macro-grid", type=int, default=2)
    parser.add_argument("--texture-micro-grid", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-samples-per-class", type=int, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=EXPERIMENT_DIR / "results" / "perturbation",
    )
    parser.add_argument("--dry-run", action="store_true", help="只运行 original 条件。")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    set_random_seed(args.seed)
    device = resolve_device(args.device)
    model, metadata = load_checkpoint(
        args.checkpoint,
        device,
        model_name=args.model_name,
        num_classes=args.num_classes,
    )
    crop_size = args.crop_size or metadata.get("crop_size")
    input_size = args.input_size or metadata.get("input_size", 224)
    full_image = bool(args.full_image or metadata.get("full_image", False))
    if not crop_size and not full_image:
        raise ValueError("无法确定物理 crop 大小，请传入 --crop-size。")
    output_dir = args.output_dir.expanduser().resolve()
    write_json(
        output_dir / "experiment_config.json",
        {
            **vars(args),
            "checkpoint_metadata": metadata,
            "crop_size_resolved": crop_size,
            "input_size_resolved": input_size,
            "full_image_resolved": full_image,
            "device_resolved": str(device),
        },
    )

    conditions = build_conditions(args)
    if args.dry_run:
        conditions = conditions[:1]
    summary = []
    original_parent_accuracy = None
    original_correctness = None
    for spec in conditions:
        name = condition_name(spec)
        print(f"评估条件：{name}")
        dataset = DiagnosticImageDataset(
            args.dataset_root,
            args.split,
            DiagnosticTransform(
                crop_size=1 if full_image else crop_size,
                input_size=input_size,
                training=False,
                view_count=args.views,
                perturbation=spec,
                seed=args.seed,
                full_image=full_image,
            ),
            DEFAULT_CLASSES,
            args.max_samples_per_class,
        )
        loader = make_loader(dataset, args.batch_size, args.num_workers, False, args.seed)
        result = evaluate_model(model, loader, dataset, device)
        if spec.name == "original":
            original_parent_accuracy = result["parent"]["accuracy"]
            original_correctness = {
                row["parent_id"]: bool(row["correct"])
                for row in result["parent_predictions"]
            }
        current_correctness = {
            row["parent_id"]: bool(row["correct"])
            for row in result["parent_predictions"]
        }
        lost_correct = gained_correct = 0
        if original_correctness is not None:
            if set(original_correctness) != set(current_correctness):
                raise RuntimeError("扰动前后原图集合不一致，不能做配对比较。")
            lost_correct = sum(
                original_correctness[key] and not current_correctness[key]
                for key in original_correctness
            )
            gained_correct = sum(
                not original_correctness[key] and current_correctness[key]
                for key in original_correctness
            )
        discordant = lost_correct + gained_correct
        if discordant:
            tail = sum(
                math.comb(discordant, index)
                for index in range(min(lost_correct, gained_correct) + 1)
            ) / (2**discordant)
            mcnemar_p = min(1.0, 2 * tail)
        else:
            mcnemar_p = 1.0
        row = metrics_row(name, result)
        row.update(
            {
                "perturbation": spec.name,
                "value": spec.value,
                "grid": spec.grid,
                "micro_grid": spec.micro_grid,
                "repeat": spec.repeat + 1,
                "parent_accuracy_drop": (
                    original_parent_accuracy - result["parent"]["accuracy"]
                    if original_parent_accuracy is not None
                    else 0.0
                ),
                "paired_lost_correct": lost_correct,
                "paired_gained_correct": gained_correct,
                "paired_mcnemar_exact_p": mcnemar_p,
            }
        )
        summary.append(row)
        condition_dir = output_dir / name
        write_json(condition_dir / "metrics.json", {"sample": result["sample"], "parent": result["parent"]})
        write_csv(condition_dir / "sample_predictions.csv", result["sample_predictions"])
        write_csv(condition_dir / "parent_predictions.csv", result["parent_predictions"])
    write_csv(output_dir / "summary.csv", summary)
    write_json(output_dir / "summary.json", summary)
    print(f"结果已保存：{output_dir}")


if __name__ == "__main__":
    main()
