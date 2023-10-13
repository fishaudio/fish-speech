import time
from datetime import timedelta
from pathlib import Path

import hydra
import torch
from lightning.fabric import Fabric
from natsort import natsorted
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
from transformers import LlamaForCausalLM
from transformers.utils import is_flash_attn_available

from speech_lm.logger import RankedLogger

# Allow TF32 on Ampere GPUs
torch.set_float32_matmul_precision("high")
torch.backends.cudnn.allow_tf32 = True

# register eval resolver
OmegaConf.register_new_resolver("eval", eval)

log = RankedLogger(__name__, rank_zero_only=True)


def train(
    model: LlamaForCausalLM,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    dataloader: torch.utils.data.DataLoader,
    global_step: int,
    fabric: Fabric,
    cfg: DictConfig,
):
    model.train()

    bar = tqdm(total=cfg.schedule.max_steps, desc="Training")
    bar.update(global_step)
    accumulate_steps = 0
    optimizer.zero_grad()
    start_time = time.time()

    while global_step < cfg.schedule.max_steps:
        for batch in dataloader:
            # Accumulate gradients
            is_accumulating = (
                accumulate_steps % cfg.schedule.gradient_accumulation_steps != 0
            )
            accumulate_steps += 1

            # Train one step
            with fabric.no_backward_sync(model, enabled=is_accumulating):
                outputs = model(**batch)
                loss = outputs.loss
                metrics = getattr(outputs, "metrics", {})
                fabric.backward(loss)

            if is_accumulating:
                continue

            # Perform gradient clipping
            grad_norm = fabric.clip_gradients(
                model, optimizer, max_norm=cfg.schedule.clip_grad_norm, norm_type=2.0
            )

            # Update
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

            fabric.log_dict(
                {
                    "train/loss": loss,
                    "train/lr": optimizer.param_groups[0]["lr"],
                    "train/grad_norm": grad_norm,
                    **{f"train/{k}": v for k, v in metrics.items()},
                },
                step=global_step,
            )

            global_step += 1
            bar.update(1)

            if global_step % cfg.schedule.log_interval == 0:
                step_time = (time.time() - start_time) / cfg.schedule.log_interval
                eta = step_time * (cfg.schedule.max_steps - global_step)
                log.info(
                    f"[{global_step}/{cfg.schedule.max_steps}] loss: {loss:.4f} "
                    + f"step time: {step_time:.2f}s "
                    f"lr: {optimizer.param_groups[0]['lr']:.2e} "
                    + f"grad_norm: {grad_norm:.2f} "
                    + f"ETA: {timedelta(round(eta))}s"
                )

                start_time = time.time()

            if global_step % cfg.schedule.save_interval == 0:
                fabric.save(
                    Path(cfg.paths.checkpoint_dir) / f"step_{global_step}.ckpt",
                    {
                        "model": model,
                        "optimizer": optimizer,
                        "scheduler": scheduler.state_dict(),
                        "global_step": global_step,
                    },
                )

            if global_step >= cfg.schedule.max_steps:
                break


@hydra.main(version_base="1.3", config_path="./configs", config_name="pretrain.yaml")
def main(cfg: DictConfig):
    log.info(f"Config: \n{OmegaConf.to_yaml(cfg)}")

    if is_flash_attn_available() is False:
        log.warning("Flash attention is not available, using default attention")

    fabric: Fabric = hydra.utils.instantiate(cfg.trainer)
    fabric.launch()
    log.info(f"Fabric: {fabric}")

    model = hydra.utils.instantiate(cfg.model)
    log.info(f"Model: {repr(model)}")

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    freeze_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    log.info(f"Trainable parameters: {trainable_params/1e6:.2f}M")
    log.info(f"Freeze parameters: {freeze_params/1e6:.2f}M")

    optimizer = hydra.utils.instantiate(cfg.optimizer, params=model.parameters())
    scheduler = hydra.utils.instantiate(cfg.scheduler, optimizer=optimizer)
    log.info(f"Optimizer: {optimizer}")
    log.info(f"Scheduler: {scheduler}")

    log.info(f"Setup fabric model & dataset")
    model = fabric.setup_module(model)
    optimizer = fabric.setup_optimizers(optimizer)

    # Build state
    global_step = 0

    # Restore training from checkpoint
    checkpoint_dir = Path(cfg.paths.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Alphabetically sort checkpoints
    checkpoints = natsorted(checkpoint_dir.glob("*.ckpt"))
    if len(checkpoints) > 0:
        checkpoint_path = checkpoints[-1]

        log.info(f"Restoring checkpoint from {checkpoint_path}")
        remainder = fabric.load(
            checkpoint_path,
            {
                "model": model,
                "optimizer": optimizer,
                "scheduler": scheduler,
            },
        )
        global_step = remainder["global_step"]
        log.info(f"Restored global step: {global_step}")

    train_dataloader = hydra.utils.instantiate(cfg.dataloader)
    log.info(f"Dataloader: {train_dataloader}")

    train_dataloader = fabric.setup_dataloaders(train_dataloader)
    log.info(f"Begin training")

    train(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        dataloader=train_dataloader,
        global_step=global_step,
        fabric=fabric,
        cfg=cfg,
    )


if __name__ == "__main__":
    main()
