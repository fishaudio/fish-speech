from typing import Optional

import hydra
import lightning as L
import torch
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig, OmegaConf

import fish_speech.utils as utils

# Allow TF32 on Ampere GPUs
torch.set_float32_matmul_precision("high")
torch.backends.cudnn.allow_tf32 = True

# register eval resolver
OmegaConf.register_new_resolver("eval", eval)

log = utils.RankedLogger(__name__, rank_zero_only=True)


@utils.task_wrapper
def train(cfg: DictConfig) -> tuple[dict, dict]:
    """Trains the model. Can additionally evaluate on a testset, using best weights obtained during
    training.
    This method is wrapped in optional @task_wrapper decorator, that controls the behavior during
    failure. Useful for multiruns, saving info about the crash, etc.
    Args:
        cfg (DictConfig): Configuration composed by Hydra.
    Returns:
        Tuple[dict, dict]: Dict with metrics and dict with all instantiated objects.
    """  # noqa: E501

    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=False)

    if cfg.get("deterministic"):
        torch.use_deterministic_algorithms(True)

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    log.info("Instantiating callbacks...")
    callbacks: list[Callback] = utils.instantiate_callbacks(cfg.get("callbacks"))

    log.info("Instantiating loggers...")
    logger: list[Logger] = utils.instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer, callbacks=callbacks, logger=logger
    )

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "callbacks": callbacks,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        utils.log_hyperparameters(object_dict)

    if cfg.get("compile"):
        log.info("Compiling model!")
        model = torch.compile(model)

    if cfg.get("train"):
        log.info("Starting training!")

        ckpt_path = cfg.get("ckpt_path")

        if ckpt_path is None:
            ckpt_path = utils.get_latest_checkpoint(cfg.paths.ckpt_dir)

        if ckpt_path is not None:
            log.info(f"Resuming from checkpoint: {ckpt_path}")

        if cfg.get("resume_weights_only"):
            log.info("Resuming weights only!")
            ckpt = torch.load(ckpt_path, map_location=model.device)
            model.load_state_dict(
                ckpt["state_dict"] if "state_dict" in ckpt else ckpt, strict=False
            )
            ckpt_path = None

        trainer.fit(model=model, datamodule=datamodule, ckpt_path=ckpt_path)

    train_metrics = trainer.callback_metrics

    if cfg.get("test"):
        log.info("Starting testing!")
        ckpt_path = trainer.checkpoint_callback.best_model_path
        if ckpt_path == "":
            log.warning("Best ckpt not found! Using current weights for testing...")
            ckpt_path = cfg.get("ckpt_path")

        trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
        log.info(f"Best ckpt path: {ckpt_path}")

    test_metrics = trainer.callback_metrics

    # merge train and test metrics
    metric_dict = {**train_metrics, **test_metrics}

    return metric_dict, object_dict


@hydra.main(
    version_base="1.3", config_path="./configs", config_name="llama_pretrain.yaml"
)
def main(cfg: DictConfig) -> Optional[float]:
    # train the model
    train(cfg)


if __name__ == "__main__":
    main()
