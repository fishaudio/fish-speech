import os
import torch
import random
import numpy as np
import time
import gc
from typing import Any, Dict, Optional, Tuple

import hydra
import lightning as L
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from lightning.pytorch.strategies import DDPStrategy
from omegaconf import DictConfig, OmegaConf

import pyrootutils
import fish_speech.utils as utils

# Set environment variables
os.environ["USE_LIBUV"] = "0"
os.environ["SLURM_NTASKS"] = os.environ.get("SLURM_NTASKS", None)
os.environ["SLURM_JOB_NAME"] = os.environ.get("SLURM_JOB_NAME", None)
os.environ["SLURM_NTASKS_PER_NODE"] = os.environ.get("SLURM_NTASKS_PER_NODE", None)

# Register eval resolver and root
pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

# Allow TF32 on Ampere GPUs
torch.set_float32_matmul_precision("high")
torch.backends.cudnn.allow_tf32 = True

# Register eval resolver
OmegaConf.register_new_resolver("eval", eval)

log = utils.RankedLogger(__name__, rank_zero_only=True)

def set_seed(seed: int):
    """Set the seed for reproducibility across various modules."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
    torch.backends.cudnn.deterministic = True  # Ensures deterministic behavior
    torch.backends.cudnn.benchmark = False  # May impact performance, but improves reproducibility
    log.info(f"Random seed set to {seed}")

def instantiate_callbacks(callbacks_cfg):
    """Helper function to instantiate callbacks from the configuration."""
    callbacks = []
    for callback_cfg in callbacks_cfg:
        callback = hydra.utils.instantiate(callback_cfg)
        callbacks.append(callback)
    return callbacks

def instantiate_components(cfg: DictConfig) -> Dict[str, Any]:
    log.info(f"Instantiating datamodule of type {cfg.data._target_}")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    log.info(f"Instantiating model of type {cfg.model._target_}")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    log.info("Instantiating callbacks...")
    callbacks: list[Callback] = instantiate_callbacks(cfg.get("callbacks", []))

    log.info("Instantiating loggers...")
    logger: list[Logger] = utils.instantiate_loggers(cfg.get("logger", []))

    log.info(f"Instantiating trainer of type {cfg.trainer._target_}")
    trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer, callbacks=callbacks, logger=logger
    )

    return {
        "datamodule": datamodule,
        "model": model,
        "callbacks": callbacks,
        "logger": logger,
        "trainer": trainer,
    }

def log_training_start(model):
    """Log the start of training and the number of trainable parameters."""
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"Starting training with {num_params} trainable parameters.")

def log_training_end(start_time):
    """Log the end of training and the total time spent."""
    elapsed_time = time.time() - start_time
    log.info(f"Training completed in {elapsed_time // 60} minutes and {elapsed_time % 60} seconds.")

def cleanup():
    """Clear cache and reset random state for memory management."""
    torch.cuda.empty_cache()
    gc.collect()

def export_model(model, export_path='model.pt'):
    """Export the model to TorchScript format."""
    model.eval()
    traced_model = torch.jit.trace(model, torch.randn(1, 3, 224, 224))  # Example input size
    traced_model.save(export_path)
    log.info(f"Model exported to {export_path}")

@utils.task_wrapper
def train(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Trains the model. Can optionally evaluate on a test set, using the best weights obtained during training."""
    # Validate the configuration
    required_keys = ["data", "model", "trainer"]
    for key in required_keys:
        if key not in cfg:
            raise ValueError(f"Missing required configuration key: {key}")

    # Set seed for reproducibility
    if cfg.get("seed"):
        set_seed(cfg.seed)

    if cfg.get("deterministic"):
        torch.use_deterministic_algorithms(True)

    # Instantiate all components
    components = instantiate_components(cfg)
    datamodule = components["datamodule"]
    model = components["model"]
    callbacks = components["callbacks"]
    logger = components["logger"]
    trainer = components["trainer"]

    # Log hyperparameters if available
    if logger:
        log.info("Logging hyperparameters!")
        utils.log_hyperparameters(components)

    # Handle training process
    if cfg.get("train"):
        start_time = time.time()
        log_training_start(model)

        ckpt_path = cfg.get("ckpt_path")
        auto_resume = False

        resume_ckpt_path = utils.get_latest_checkpoint(cfg.paths.ckpt_dir)
        if resume_ckpt_path is not None:
            ckpt_path = resume_ckpt_path
            auto_resume = True

        if ckpt_path is not None:
            log.info(f"Resuming from checkpoint: {ckpt_path}")

        if cfg.get("resume_weights_only") and not auto_resume:
            log.info("Resuming weights only!")
            try:
                ckpt = torch.load(ckpt_path, map_location=model.device)
                if "state_dict" in ckpt:
                    ckpt = ckpt["state_dict"]
                model.load_state_dict(ckpt, strict=False)
                log.info("Successfully loaded model weights.")
            except Exception as e:
                log.error(f"Error loading checkpoint: {e}")
                raise
            ckpt_path = None

        trainer.fit(model=model, datamodule=datamodule, ckpt_path=ckpt_path)

        log_training_end(start_time)

    train_metrics = trainer.callback_metrics

    # Handle testing process
    if cfg.get("test"):
        log.info("Starting testing phase...")
        ckpt_path = cfg.get("test_ckpt_path", trainer.checkpoint_callback.best_model_path)
        if not ckpt_path:
            log.warning("No checkpoint provided for testing. Using current model weights...")
            ckpt_path = None  # Use current model weights

        trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
        log.info(f"Testing completed with checkpoint: {ckpt_path}")

    test_metrics = trainer.callback_metrics

    # Merge train and test metrics
    metric_dict = {**train_metrics, **test_metrics}

    # Export model after training if enabled
    if cfg.get("export_model"):
        export_path = cfg.get("export_model_path", "model.pt")
        export_model(model, export_path)

    # Cleanup resources
    cleanup()

    return metric_dict, components

@hydra.main(version_base="1.3", config_path="./configs", config_name="llama_pretrain.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    # Train the model
    train(cfg)

if __name__ == "__main__":
    main()
