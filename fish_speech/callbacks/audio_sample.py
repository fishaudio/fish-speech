from pathlib import Path
from typing import Optional

import torch
import torchaudio
from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback
from loguru import logger


class AudioSampleCallback(Callback):
    """
    Generates a sample wav file whenever a checkpoint is saved.
    Reuses the training model (no second copy loaded). Before inference,
    optimizer states are offloaded to CPU and gradients are freed to make
    room for KV caches and the codec model.
    """

    def __init__(
        self,
        text: str,
        codec_checkpoint_path: str,
        reference_audio_path: Optional[str] = None,
        reference_text: Optional[str] = None,
        output_dir: Optional[str] = None,
        audio_format: str = "wav",
    ):
        super().__init__()
        self.text = text
        self.codec_checkpoint_path = codec_checkpoint_path
        self.reference_audio_path = reference_audio_path
        self.reference_text = reference_text
        self.output_dir = Path(output_dir) if output_dir else None
        self.audio_format = audio_format
        self._codec = None
        self._cached_prompt_tokens = None  # encoded reference, kept on CPU
        self._last_sampled_step = -1

    # ------------------------------------------------------------------
    # Memory management: free training-only GPU allocations for inference
    # ------------------------------------------------------------------

    @staticmethod
    def _offload_optimizer(trainer):
        """Move optimizer states to CPU to free GPU memory."""
        for optimizer in trainer.optimizers:
            optimizer.zero_grad(set_to_none=True)
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor) and v.is_cuda:
                        state[k] = v.cpu()

    @staticmethod
    def _restore_optimizer(trainer, device):
        """Move optimizer states back to GPU."""
        for optimizer in trainer.optimizers:
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor) and not v.is_cuda:
                        state[k] = v.to(device)

    # ------------------------------------------------------------------
    # Codec / reference encoding
    # ------------------------------------------------------------------

    def _ensure_codec(self, device):
        if self._codec is not None:
            return
        from fish_speech.models.text2semantic.inference import load_codec_model

        logger.info("AudioSampleCallback: loading codec model...")
        self._codec = load_codec_model(self.codec_checkpoint_path, device)
        logger.info("AudioSampleCallback: codec loaded.")

    def _free_codec(self):
        del self._codec
        self._codec = None

    @torch.no_grad()
    def _get_prompt_tokens(self, device):
        """Return cached reference tokens, encoding on first call."""
        if not self.reference_audio_path:
            return None

        if self._cached_prompt_tokens is not None:
            return self._cached_prompt_tokens

        from fish_speech.models.text2semantic.inference import encode_audio

        self._ensure_codec(device)
        codes = encode_audio(self.reference_audio_path, self._codec, device)
        self._cached_prompt_tokens = codes.cpu()  # keep on CPU
        return self._cached_prompt_tokens

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _generate_sample(self, trainer: Trainer, pl_module: LightningModule):
        from fish_speech.models.text2semantic.inference import (
            decode_one_token_ar,
            decode_to_audio,
            generate_long,
        )

        device = pl_module.device
        step = trainer.global_step

        # Free training memory to make room for KV caches + codec
        self._offload_optimizer(trainer)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        model = pl_module.model
        was_training = model.training
        model.eval()

        try:
            prompt_tokens = self._get_prompt_tokens(device)
            self._ensure_codec(device)

            results = list(
                generate_long(
                    model=model,
                    device=device,
                    decode_one_token=decode_one_token_ar,
                    text=self.text,
                    num_samples=1,
                    top_p=0.8,
                    temperature=0.8,
                    repetition_penalty=1.1,
                    prompt_text=(
                        self.reference_text
                        if prompt_tokens is not None
                        else None
                    ),
                    prompt_tokens=prompt_tokens,
                )
            )

            # Collect codes from all "sample" responses and concatenate
            code_chunks = [r.codes for r in results if r.codes is not None]
            if not code_chunks:
                logger.warning(
                    "AudioSampleCallback: generate_long returned no samples."
                )
                return

            codes = (
                torch.cat(code_chunks, dim=1)
                if len(code_chunks) > 1
                else code_chunks[0]
            )

            # Decode codes to audio with the codec
            codes = codes.to(device=device)
            wav = decode_to_audio(codes, self._codec)
            wav = wav.float().cpu()

            # Determine output path
            if self.output_dir:
                out_dir = self.output_dir
            else:
                out_dir = Path(trainer.log_dir) / "audio_samples"
            out_dir.mkdir(parents=True, exist_ok=True)

            sample_rate = self._codec.sample_rate
            out_path = out_dir / f"step_{step:09d}.{self.audio_format}"
            torchaudio.save(str(out_path), wav.unsqueeze(0), sample_rate, format=self.audio_format)
            logger.info(f"AudioSampleCallback: saved sample to {out_path}")

            # Log to TensorBoard if available
            for lg in trainer.loggers:
                if hasattr(lg, "experiment") and hasattr(
                    lg.experiment, "add_audio"
                ):
                    lg.experiment.add_audio(
                        "audio_sample",
                        wav,
                        global_step=step,
                        sample_rate=sample_rate,
                    )

        except Exception as e:
            logger.exception(
                f"AudioSampleCallback: failed to generate sample: {e}"
            )
        finally:
            self._teardown_caches(model)
            self._free_codec()
            if was_training:
                model.train()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self._restore_optimizer(trainer, device)

    # ------------------------------------------------------------------
    # Cache teardown
    # ------------------------------------------------------------------

    @staticmethod
    def _teardown_caches(model):
        """Reset KV caches that were created during inference back to None."""
        for layer in model.layers:
            layer.attention.kv_cache = None

        if hasattr(model, "fast_layers"):
            for layer in model.fast_layers:
                layer.attention.kv_cache = None

        model._cache_setup_done = False
        model.max_seq_len = -1
        model.max_batch_size = -1

    # ------------------------------------------------------------------
    # Hook
    # ------------------------------------------------------------------

    def _checkpoint_interval(self, trainer) -> int:
        """Return every_n_train_steps from ModelCheckpoint, or 0 if not found."""
        from lightning.pytorch.callbacks import ModelCheckpoint

        for cb in trainer.callbacks:
            if isinstance(cb, ModelCheckpoint) and cb._every_n_train_steps:
                return cb._every_n_train_steps
        return 0

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if trainer.global_rank != 0:
            return
        step = trainer.global_step
        if step <= 0 or step == self._last_sampled_step:
            return
        interval = self._checkpoint_interval(trainer)
        if interval <= 0 or step % interval != 0:
            return
        self._last_sampled_step = step
        self._generate_sample(trainer, pl_module)
