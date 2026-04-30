import os
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import hydra
import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf, open_dict
from pytorch_lightning import Trainer
from transformers import AutoProcessor, AutoTokenizer

from simlingo_training.config import TrainConfig
from simlingo_training.utils.logging_project import setup_logging
# from simlingo_training.callbacks.visualise import VisualiseCallback


def _resolve_repo_path(path_value):
    if path_value is None:
        return None

    path = Path(str(path_value)).expanduser()
    if not path.is_absolute():
        path = (repo_root / path).resolve()
    return path


def _find_run_dir_from_checkpoint(checkpoint_path: Path) -> Path:
    candidates = [
        checkpoint_path,
        checkpoint_path.parent,
        checkpoint_path.parent.parent,
    ]
    for candidate in candidates:
        if (candidate / ".hydra" / "config.yaml").exists():
            return candidate
    return checkpoint_path.parent.parent if checkpoint_path.is_file() else checkpoint_path


def _copy_eval_overrides(src_cfg, dst_cfg, load_path):
    with open_dict(dst_cfg):
        dst_cfg.eval_mode = getattr(src_cfg, "eval_mode", "Dreaming")
        dst_cfg.eval_load_path = str(load_path) if load_path is not None else None
        dst_cfg.eval_batch_size = getattr(src_cfg, "eval_batch_size", 1)
        dst_cfg.eval_num_workers = getattr(src_cfg, "eval_num_workers", 0)
        dst_cfg.limit_predict_batches = getattr(src_cfg, "limit_predict_batches", None)
        dst_cfg.gpus = getattr(src_cfg, "gpus", 1)
        dst_cfg.precision = getattr(src_cfg, "precision", dst_cfg.precision)
        dst_cfg.strategy = getattr(src_cfg, "strategy", dst_cfg.strategy)
        dst_cfg.enable_wandb = getattr(src_cfg, "enable_wandb", False)
        dst_cfg.checkpoint = None if load_path is not None else getattr(src_cfg, "checkpoint", None)

    dst_cfg.data_module.qa_dataset = src_cfg.data_module.qa_dataset
    dst_cfg.data_module.insteval_dataset = src_cfg.data_module.insteval_dataset
    dst_cfg.data_module.batch_size = getattr(src_cfg, "eval_batch_size", 1)
    dst_cfg.data_module.num_workers = getattr(src_cfg, "eval_num_workers", 0)

    with open_dict(dst_cfg.data_module.base_dataset):
        for key, value in src_cfg.data_module.base_dataset.items():
            dst_cfg.data_module.base_dataset[key] = value

    return dst_cfg


@hydra.main(config_path=f"config", config_name="config", version_base="1.1")
def main(cfg: TrainConfig):
    torch.set_float32_matmul_precision("high")
    pl.seed_everything(cfg.seed, workers=True)

    requested_cfg = cfg
    load_path = _resolve_repo_path(getattr(cfg, "eval_load_path", None))
    if load_path is not None:
        if not load_path.exists():
            raise FileNotFoundError(f"eval_load_path does not exist: {load_path}")

        run_dir = _find_run_dir_from_checkpoint(load_path)
        load_path_config = run_dir / ".hydra" / "config.yaml"
        if load_path_config.exists():
            cfg = OmegaConf.load(load_path_config)
            cfg = _copy_eval_overrides(requested_cfg, cfg, load_path)

    eval_mode = getattr(cfg, "eval_mode", "Dreaming")

    print(f'Eval mode: {eval_mode}')
    print(f'Checkpoint: {load_path}')
    print(f"Using {cfg.gpus} GPUs")
    
    if eval_mode == "QA" or eval_mode == "commentary":
        cfg.data_module.dreamer_dataset = None
        cfg.data_module.driving_dataset = None
        cfg.data_module.insteval_dataset = None 
    elif eval_mode == "Dreaming":
        cfg.data_module.dreamer_dataset = None
        cfg.data_module.driving_dataset = None
        cfg.data_module.qa_dataset = None
    
    if eval_mode == "QA":
        cfg.data_module.base_dataset.use_commentary = False
        cfg.data_module.base_dataset.use_qa = True
    elif eval_mode == "commentary":
        cfg.data_module.base_dataset.use_commentary = True
        cfg.data_module.base_dataset.use_qa = False
    elif eval_mode == "Dreaming":
        cfg.data_module.base_dataset.use_safety_flag = True
    
    # disable image augmentation
    cfg.data_module.base_dataset.img_augmentation = False
    
    # disable img_shift_augmentation
    cfg.data_module.base_dataset.img_shift_augmentation = False
    
    if "2B" in cfg.model.language_model.variant:
        processor = AutoTokenizer.from_pretrained(cfg.model.language_model.variant, trust_remote_code=True, use_fast=False)
    else:
        processor = AutoProcessor.from_pretrained(cfg.model.language_model.variant, trust_remote_code=True, use_fast=False)
    model_type_name = cfg.model.vision_model.variant.split('/')[1]
    cache_dir = f"pretrained/{(model_type_name)}"
    
    data_module = hydra.utils.instantiate(
        cfg.data_module, 
        processor=processor,
        encoder_variant=cfg.model.vision_model.variant,
        llm_variant=cfg.model.language_model.variant,
        predict=True,
        _recursive_=False
    )
    
    model = hydra.utils.instantiate(
        cfg.model,
        cfg_data_module=cfg.data_module,
        processor=processor,
        cache_dir=cache_dir,
        _recursive_=False
        )

    if cfg.checkpoint is not None:
        if os.path.isdir(cfg.checkpoint):
            from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint

            state_dict = get_fp32_state_dict_from_zero_checkpoint(cfg.checkpoint)
        else:
            state_dict = torch.load(cfg.checkpoint, map_location="cpu")
        model.load_state_dict(state_dict)

        
    # print config
    print(OmegaConf.to_yaml(cfg))
    os.environ["WANDB_DISABLE_CODE"] = "True"

    
    # setup logging
    setup_logging(cfg)

    # resume training
    resume_path = "./checkpoints/last.ckpt"


    if os.path.exists(resume_path) and cfg.resume:
        resume_path = resume_path
    else:
        resume_path = None
    
    # setup lightning logger
    loggers = []

    strategy = cfg.strategy
    if strategy == "deepspeed_stage_2":
        strategy = pl.strategies.DeepSpeedStrategy(
            stage=2, loss_scale=cfg.fp16_loss_scale, logging_batch_size_per_gpu=cfg.data_module.batch_size
        )
  
    print(f"Number of GPUS: {cfg.gpus}")
    overfit = 0
    
    trainer_kwargs = {
        "benchmark": True,
        "gradient_clip_val": 0.3,
        "log_every_n_steps": cfg.log_every_n_steps,
        "logger": loggers,
        "precision": cfg.precision,
        "max_epochs": cfg.max_epochs,
        "overfit_batches": overfit,
        "check_val_every_n_epoch": cfg.val_every_n_epochs,
    }
    if cfg.limit_predict_batches is not None:
        trainer_kwargs["limit_predict_batches"] = cfg.limit_predict_batches

    if cfg.gpus >= 1:
        trainer_kwargs.update({"accelerator": "gpu", "devices": cfg.gpus})
        if strategy not in {None, "auto", "none"}:
            trainer_kwargs["strategy"] = strategy
            trainer_kwargs["sync_batchnorm"] = True
    else:
        trainer_kwargs.update({"accelerator": "cpu", "devices": 1})

    trainer = Trainer(**trainer_kwargs)

    trainer.predict(
        model,
        data_module,
        ckpt_path=str(load_path) if load_path is not None else None,
        weights_only=False,
    )

if __name__ == "__main__":
    main()
