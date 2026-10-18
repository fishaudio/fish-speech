import random

import numpy as np
import torch


def set_seed(seed: int):
    if seed < 0:
        seed = -seed
    if seed > (1 << 31):
        seed = 1 << 31

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
