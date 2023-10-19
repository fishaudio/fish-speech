import time
from collections import defaultdict
from datetime import timedelta
from pathlib import Path
from typing import Optional

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


def valid(
    model: LlamaForCausalLM,
    valid_dataloader: Optional[torch.utils.data.DataLoader],
    global_step: int,
    fabric: Fabric,
    cfg: DictConfig,
):
    model.eval()
    log.info(f"Evaluating at step {global_step}")

    accumulate_infos = None

    for idx, batch in tqdm(enumerate(valid_dataloader), desc="Evaluating"):
        outputs = model(**batch)
        loss = outputs.loss
        metrics = getattr(outputs, "metrics", {})
        log_info = {
            "valid/loss": float(loss),
            **{f"valid/{k}": float(v) for k, v in metrics.items()},
        }

        fabric.log_dict(
            log_info,
            step=global_step + idx,
        )

        # Update log info
        if accumulate_infos is None:
            accumulate_infos = log_info
        else:
            assert set(accumulate_infos.keys()) == set(
                log_info.keys()
            ), "Log keys changed during evaluation"
            for k in accumulate_infos.keys():
                accumulate_infos[k] += log_info[k]

        if idx == getattr(cfg.schedule, "eval_max_batches", None):
            break

    # Log average
    items = []
    for k in accumulate_infos.keys():
        items.append(f"{k}: {accumulate_infos[k] / (idx + 1):.4f}")
    log.info(f"Average: {' | '.join(items)}")


def train(
    model: LlamaForCausalLM,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    train_dataloader: torch.utils.data.DataLoader,
    valid_dataloader: Optional[torch.utils.data.DataLoader],
    global_step: int,
    fabric: Fabric,
    cfg: DictConfig,
):
    bar = tqdm(total=cfg.schedule.max_steps, desc="Training")
    bar.update(global_step)
    accumulate_steps = 0
    optimizer.zero_grad()

    # Start time is ~model forward time + data loading time
    start_time = time.time()
    trackers = defaultdict(list)

    while global_step < cfg.schedule.max_steps:
        last_batch_time = time.time()
        for batch in train_dataloader:
            # Measure time used by data loading
            trackers["data_time"].append(time.time() - last_batch_time)

            # Measure time used by model forward
            model_begin_time = time.time()
            model.train()

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

                # Need to divide loss by accumulation steps
                fabric.backward(loss / cfg.schedule.gradient_accumulation_steps)

                # Update trackers
                trackers["loss"].append(float(loss))
                trackers["lr"].append(float(optimizer.param_groups[0]["lr"]))
                for k, v in metrics.items():
                    trackers[f"metrics/{k}"].append(float(v))

            trackers["model_time"].append(time.time() - model_begin_time)

            if is_accumulating:
                last_batch_time = time.time()
                continue

            # Check all trackers has the same length
            assert (
                len(set(len(v) for k, v in trackers.items() if k != "grad_norm")) == 1
            ), "Trackers has ambiguous length"

            # Perform gradient clipping
            grad_norm = fabric.clip_gradients(
                model,
                optimizer,
                max_norm=cfg.schedule.clip_grad_norm,
                norm_type=2.0,
            )

            # We can't average gradients across multiple steps
            trackers["grad_norm"].append(float(grad_norm))

            # Update
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

            fabric.log_dict(
                {
                    f"train/{k}": sum(v[-accumulate_steps:])
                    / len(v[-accumulate_steps:])
                    for k, v in trackers.items()
                },
                step=global_step,
            )

            global_step += 1
            bar.update(1)

            if global_step % cfg.schedule.log_interval == 0:
                step_time = (time.time() - start_time) / cfg.schedule.log_interval
                eta = step_time * (cfg.schedule.max_steps - global_step)
                additional_info = [
                    f"{k}: {sum(v[-cfg.schedule.log_interval:]) / len(v[-cfg.schedule.log_interval:]):.4f}"
                    for k, v in trackers.items()
                    if k != "lr"  # lr use .2e format
                ]

                log.info(
                    f"[{global_step}/{cfg.schedule.max_steps}] "
                    + f"step_time: {step_time:.2f}s "
                    + f"ETA: {timedelta(seconds=round(eta))}s "
                    f"lr: {optimizer.param_groups[0]['lr']:.2e} "
                    + " ".join(additional_info)
                )

                # Reset trackers
                trackers = defaultdict(list)

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

            if (
                getattr(cfg.schedule, "eval_interval", None) is not None
                and global_step % cfg.schedule.eval_interval == 0
                and valid_dataloader is not None
            ):
                valid(model, valid_dataloader, global_step, fabric, cfg)

            if global_step >= cfg.schedule.max_steps:
                break

            last_batch_time = time.time()


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

    train_dataloader = hydra.utils.instantiate(cfg.train_dataloader)
    log.info(f"Train Dataloader: {train_dataloader}")

    valid_dataloader = None
    if getattr(cfg, "valid_dataloader", None) is not None:
        valid_dataloader = hydra.utils.instantiate(cfg.valid_dataloader)
        log.info(f"Valid Dataloader: {valid_dataloader}")

    train_dataloader = fabric.setup_dataloaders(train_dataloader)
    if valid_dataloader is not None:
        valid_dataloader = fabric.setup_dataloaders(valid_dataloader)

    log.info(f"Begin training")

    train(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_dataloader=train_dataloader,
        valid_dataloader=valid_dataloader,
        global_step=global_step,
        fabric=fabric,
        cfg=cfg,
    )


if __name__ == "__main__":
    main()
