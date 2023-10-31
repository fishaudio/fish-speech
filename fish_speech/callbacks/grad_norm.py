from typing import Union

import lightning.pytorch as pl
import torch
from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback
from torch import Tensor
from torch.utils._foreach_utils import (
    _group_tensors_by_device_and_dtype,
    _has_foreach_support,
)


@torch.no_grad()
def grad_norm(
    parameters: Union[Tensor, list[Tensor]],
    norm_type: float = 2.0,
) -> float:
    """
    Returns the norm of the gradients of the given parameters.

    Args:
        parameters (Iterable[Tensor] or Tensor): an iterable of Tensors or a
            single Tensor that will have gradients normalized
        norm_type (float): type of the used p-norm.

    Returns:
        Total norm of the parameter gradients (viewed as a single vector).
    """  # noqa: E501

    if isinstance(parameters, Tensor):
        parameters = [parameters]

    grads = [p.grad for p in parameters if p.grad is not None]
    first_device = grads[0].device
    grouped_grads: dict[
        tuple[torch.device, torch.dtype], list[list[Tensor]]
    ] = _group_tensors_by_device_and_dtype(
        [[g.detach() for g in grads]]
    )  # type: ignore[assignment]

    norms = []
    for (device, _), ([grads], _) in grouped_grads.items():
        if _has_foreach_support(grads, device=device):
            norms.extend(torch._foreach_norm(grads, norm_type))
        else:
            norms.extend([torch.norm(g, norm_type) for g in grads])

    return torch.norm(torch.stack([norm.to(first_device) for norm in norms]), norm_type)


class GradNormMonitor(Callback):
    """
    Callback that computes the gradient norm of the model parameters.
    """

    def __init__(self, norm_type: float = 2.0, logging_interval: str = "step") -> None:
        """
        Args:
            norm_type (float): type of the used p-norm.
            logging_interval (str): "step" or "epoch".
        """
        super().__init__()
        self.norm_type = norm_type
        self.logging_interval = logging_interval

    def on_after_backward(self, trainer: Trainer, model: LightningModule) -> None:
        """
        Computes the gradient norm of the model parameters and logs it to the logger.

        Args:
            trainer (Trainer): The trainer object
            model (LightningModule): The current lightningModule
        """

        grad_norm_val = grad_norm(model.parameters(), self.norm_type)

        model_name = model.__class__.__name__.lower()

        on_step = self.logging_interval == "step"
        model.log(
            f"train/{model_name}/grad_norm",
            grad_norm_val,
            on_step=on_step,
            on_epoch=not on_step,
        )
