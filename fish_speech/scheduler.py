import math


def get_cosine_schedule_with_warmup_lr_lambda(
    current_step: int,
    *,
    num_warmup_steps: int | float,
    num_training_steps: int,
    num_cycles: float = 0.5,
    final_lr_ratio: float = 0.0,
):
    if 0 < num_warmup_steps < 1:  # float mode
        num_warmup_steps = int(num_warmup_steps * num_training_steps)

    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))

    progress = float(current_step - num_warmup_steps) / float(
        max(1, num_training_steps - num_warmup_steps)
    )

    return max(
        final_lr_ratio,
        0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)),
    )


def get_constant_schedule_with_warmup_lr_lambda(
    current_step: int,
    *,
    num_warmup_steps: int | float,
    num_training_steps: int | None = None,
):
    if 0 < num_warmup_steps < 1:  # float mode
        num_warmup_steps = int(num_warmup_steps * num_training_steps)

    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))

    return 1.0
