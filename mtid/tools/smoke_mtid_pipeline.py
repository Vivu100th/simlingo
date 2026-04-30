"""Reusable smoke test for the MTID training data path.

The default mode stops after DataModule collation, which verifies dataset
discovery, image preprocessing, prompt tokenization, and batch tensor shapes
without loading the full InternVL model weights.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import hydra
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from transformers import AutoProcessor

import simlingo_training.config  # noqa: F401 - registers Hydra structured configs
import simlingo_training.dataloader.dataset_base as dataset_base


def shape_of(value) -> tuple[int, ...]:
    return tuple(value.shape)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke test the MTID DataModule path.")
    parser.add_argument("--experiment", default="mtid_debug")
    parser.add_argument(
        "--dataset-mode",
        choices=["dreamer", "driving", "config"],
        default="dreamer",
        help="Which dataset branch to keep for the smoke test.",
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--with-model-loss", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_base.get_original_cwd = lambda: str(Path.cwd())

    overrides = [
        f"experiment={args.experiment}",
        f"data_module.batch_size={args.batch_size}",
        f"data_module.num_workers={args.num_workers}",
        "data_module.base_dataset.img_augmentation=false",
        "data_module.base_dataset.img_shift_augmentation=false",
    ]
    if args.dataset_mode == "dreamer":
        overrides.append("data_module.driving_dataset=null")
    elif args.dataset_mode == "driving":
        overrides.append("data_module.dreamer_dataset=null")

    config_dir = Path.cwd() / "simlingo_training" / "config"
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg = compose(config_name="config", overrides=overrides)

    processor = AutoProcessor.from_pretrained(cfg.model.vision_model.variant, trust_remote_code=True)
    data_module = hydra.utils.instantiate(
        cfg.data_module,
        processor=processor,
        encoder_variant=cfg.model.vision_model.variant,
        llm_variant=cfg.model.language_model.variant,
        _recursive_=False,
    )
    data_module.setup()

    train_batch = next(iter(data_module.train_dataloader()))
    val_batch = next(iter(data_module.val_dataloader()))

    print("experiment", args.experiment)
    print("dataset_mode", args.dataset_mode)
    print("dreamer_folder", cfg.data_module.base_dataset.dreamer_folder)
    print("train_dataset_len", len(data_module.train_dataset))
    print("val_dataset_len", len(data_module.val_dataset))
    print("train_camera_images", shape_of(train_batch.driving_input.camera_images))
    print("train_waypoints", shape_of(train_batch.driving_label.waypoints))
    print("train_path", shape_of(train_batch.driving_label.path))
    print("val_camera_images", shape_of(val_batch.driving_input.camera_images))
    print("val_waypoints", shape_of(val_batch.driving_label.waypoints))
    print("val_path", shape_of(val_batch.driving_label.path))

    if args.with_model_loss:
        model = hydra.utils.instantiate(
            cfg.model,
            cfg_data_module=cfg.data_module,
            processor=processor,
            cache_dir=None,
            _recursive_=False,
        )
        output, _loss_logs = model.forward_loss(train_batch)
        print("train_loss", float(output.loss.detach().cpu()))


if __name__ == "__main__":
    main()
