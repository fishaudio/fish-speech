import hydra
import torch
from loguru import logger
from omegaconf import DictConfig, OmegaConf

# register eval resolver
OmegaConf.register_new_resolver("eval", eval)


@hydra.main(
    version_base="1.3",
    config_path="../../fish_speech/configs",
    config_name="hubert_vq.yaml",
)
def main(cfg: DictConfig):
    generator_ckpt = cfg.get(
        "generator_ckpt", "results/hubert-vq-pretrain/rcell/G_23000.pth"
    )
    discriminator_ckpt = cfg.get(
        "discriminator_ckpt", "results/hubert-vq-pretrain/rcell/D_23000.pth"
    )
    model = hydra.utils.instantiate(cfg.model)

    # Generator
    logger.info(f"Model loaded, restoring from {generator_ckpt}")
    generator_weights = torch.load(generator_ckpt, map_location="cpu")["model"]

    # Decoder
    generator_state = {
        k[4:]: v for k, v in generator_weights.items() if k.startswith("dec.")
    }
    logger.info(f"Found {len(generator_state)} HiFiGAN weights, restoring...")
    model.generator.load_state_dict(generator_state, strict=True)
    logger.info("Generator weights restored.")

    # Posterior Encoder
    encoder_state = {
        k[6:]: v for k, v in generator_weights.items() if k.startswith("enc_q.")
    }
    logger.info(f"Found {len(encoder_state)} posterior encoder weights, restoring...")
    model.posterior_encoder.load_state_dict(encoder_state, strict=True)
    logger.info("Posterior encoder weights restored.")

    # Flow
    # flow_state = {
    #     k[5:]: v for k, v in generator_weights.items() if k.startswith("flow.")
    # }
    # logger.info(f"Found {len(flow_state)} flow weights, restoring...")
    # model.flow.load_state_dict(flow_state, strict=True)
    # logger.info("Flow weights restored.")

    # Discriminator
    logger.info(f"Model loaded, restoring from {discriminator_ckpt}")
    discriminator_weights = torch.load(discriminator_ckpt, map_location="cpu")["model"]
    logger.info(
        f"Found {len(discriminator_weights)} discriminator weights, restoring..."
    )
    model.discriminator.load_state_dict(discriminator_weights, strict=True)
    logger.info("Discriminator weights restored.")

    # Restore kmeans
    logger.info("Reset vq projection layer to mimic avg pooling")
    torch.nn.init.normal_(
        model.semantic_encoder.in_proj.weight,
        mean=1
        / (
            model.semantic_encoder.in_proj.weight.shape[0]
            * model.semantic_encoder.in_proj.weight.shape[-1]
        ),
        std=1e-2,
    )
    model.semantic_encoder.in_proj.bias.data.zero_()

    kmeans_ckpt = "results/hubert-vq-pretrain/kmeans.pt"
    kmeans_ckpt = torch.load(kmeans_ckpt, map_location="cpu")

    centroids = kmeans_ckpt["centroids"][0]
    bins = kmeans_ckpt["bins"][0]
    logger.info(
        f"Restoring kmeans centroids with shape {centroids.shape} and bins {bins.shape}"
    )

    state_dict = {
        "_codebook.inited": torch.Tensor([True]),
        "_codebook.cluster_size": bins,
        "_codebook.embed": centroids,
        "_codebook.embed_avg": centroids.clone(),
    }

    model.semantic_encoder.vq.load_state_dict(state_dict, strict=True)

    torch.save(model.state_dict(), cfg.ckpt_path)
    logger.info("Done")


if __name__ == "__main__":
    main()
