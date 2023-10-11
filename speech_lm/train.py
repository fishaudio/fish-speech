import torch
from lightning.fabric import Fabric
import hydra
from omegaconf import DictConfig, OmegaConf
import pyrootutils

# Allow TF32 on Ampere GPUs
torch.set_float32_matmul_precision("high")
torch.backends.cudnn.allow_tf32 = True

# register eval resolver and root
pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
OmegaConf.register_new_resolver("eval", eval)

# flake8: noqa: E402
from speech_lm.dataset import build_dataset

@hydra.main(version_base="1.3", config_path="./configs", config_name="pretrain.yaml")
def main(cfg: DictConfig):
    print(cfg)

if __name__ == "__main__":
    main()
