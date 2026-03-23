from lightning.pytorch.callbacks import TQDMProgressBar


class GradAccumProgressBar(TQDMProgressBar):
    """
    Progress bar that accounts for gradient accumulation so the total
    reflects actual forward passes rather than optimizer steps.
    """

    @property
    def total_train_batches(self):
        total = super().total_train_batches
        accumulate = self.trainer.accumulate_grad_batches
        if isinstance(total, int) and accumulate > 1:
            return total * accumulate
        return total
