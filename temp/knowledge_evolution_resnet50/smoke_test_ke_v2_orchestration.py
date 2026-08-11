"""Data-training-free integration smoke test for the KE-V2 generation manager."""

from __future__ import annotations

import copy
import gc
import logging
import tempfile
from pathlib import Path

import torch

import run_knowledge_evolution as runner
from ke_core import create_kels_masks, reset_reset_hypothesis_from_fresh_model


def main() -> None:
    common = runner._load_yaml_mapping(runner.DEFAULT_COMMON_CONFIG, "Common config")
    model_config = runner.train_batch.load_model_config(runner.DEFAULT_MODEL_CONFIG)
    model_config = copy.deepcopy(model_config)
    model_config["model"]["backbone"]["pretrained"] = False
    dataset_root = runner._resolve_project_path(common["dataset_root"])
    device = torch.device("cpu")

    with tempfile.TemporaryDirectory(prefix="ke_v2_smoke_", dir=runner.EXPERIMENT_DIR) as temp:
        output_dir = Path(temp)
        config = runner._build_runtime_config(
            common,
            model_config,
            runner.DEFAULT_MODEL_CONFIG,
            dataset_root,
            output_dir,
            device,
            epochs=3,
            run_name="ke_v2_orchestration_smoke",
        )
        trainer = runner.KEV2Trainer(
            config=config,
            device=device,
            generation=1,
            split_rate=0.9,
            transition_source="previous_generation_best_validation",
        )
        checkpoint_hook = next(
            hook
            for hook in trainer.hook_manager.hooks
            if isinstance(hook, runner.LexicographicBestCheckpointHook)
        )

        trainer.last_full_validation_metrics = {"macro_f1": 0.70}
        checkpoint_hook.on_epoch_end(
            trainer,
            0,
            {
                "train_acc": 0.74,
                "val_acc": 0.75,
                "val_qwk": 0.92,
                "val_loss": 0.80,
                "val_mae": 0.30,
            },
        )
        trainer.last_full_validation_metrics = {"macro_f1": 0.71}
        checkpoint_hook.on_epoch_end(
            trainer,
            1,
            {
                "train_acc": 0.76,
                "val_acc": 0.75,
                "val_qwk": 0.93,
                "val_loss": 0.90,
                "val_mae": 0.29,
            },
        )
        # Same accuracy/QWK but worse loss must not replace epoch 2.
        checkpoint_hook.on_epoch_end(
            trainer,
            2,
            {
                "train_acc": 0.80,
                "val_acc": 0.75,
                "val_qwk": 0.93,
                "val_loss": 0.95,
                "val_mae": 0.28,
            },
        )
        best_path = output_dir / "best_val.pth"
        checkpoint = runner.train_batch.load_checkpoint(best_path, device)
        assert checkpoint["epoch_number"] == 2
        assert checkpoint["val_qwk"] == 0.93
        assert "model" in checkpoint and "model_state_dict" in checkpoint
        with runner._temporary_best_model_alias(output_dir):
            assert (output_dir / "best_model.pth").is_file()
        assert not (output_dir / "best_model.pth").exists()

        runner._load_best_checkpoint_into_model(trainer, best_path)
        masks = create_kels_masks(trainer.model, split_rate=0.9)
        fresh_model = runner._build_fresh_random_model(trainer, reset_seed=12028)
        report = reset_reset_hypothesis_from_fresh_model(
            trainer.model,
            fresh_model,
            masks,
        )
        assert report["max_fit_abs_difference"] == 0.0
        assert report["reset_changed_fraction"] > 0.99

        old_optimizer_id = id(trainer.optimizer)
        old_scheduler_id = id(trainer.scheduler)
        trainer.restart_optimizer_and_scheduler(total_epochs=3)
        assert id(trainer.optimizer) != old_optimizer_id
        assert id(trainer.scheduler) != old_scheduler_id

        del checkpoint, fresh_model, trainer
        gc.collect()
        logging.shutdown()

    print("KE-V2 orchestration smoke test passed.")


if __name__ == "__main__":
    main()
