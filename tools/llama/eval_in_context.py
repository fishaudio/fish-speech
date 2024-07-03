import pyrootutils
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from transformers import AutoTokenizer

# register eval resolver and root
pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from torch.utils.data import DataLoader

from fish_speech.datasets.semantic import AutoAugTextDataset, TextDataCollator
from tools.llama.generate import load_model


def smooth(
    scalars: list[float], weight: float
) -> list[float]:  # Weight between 0 and 1
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)  # Save it
        last = smoothed_val  # Anchor the last smoothed value

    return smoothed


@torch.inference_mode()
def analyze_one_model(loader, config, weight, max_length):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(
        config,
        weight,
        device,
        torch.bfloat16,
        max_length,
        compile=False,
    )[0]

    current_step = 0
    model.eval()

    semantic_loss_sum = torch.zeros(
        max_length,
        dtype=torch.float32,
        device=device,
    )
    counter = torch.zeros(
        max_length,
        dtype=torch.long,
        device=device,
    )

    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}

        labels = batch["labels"]
        outputs = model(
            inp=batch["inputs"],
            key_padding_mask=batch["attention_masks"],
        )

        token_logits = outputs.token_logits
        codebook_logits = outputs.codebook_logits

        # Generate labels
        base_loss = F.cross_entropy(
            token_logits.reshape(-1, token_logits.size(-1)),
            labels[:, 0].reshape(-1),
            ignore_index=-100,
            reduction="none",
        )

        codebook_labels = labels[:, 1 : 1 + model.config.num_codebooks].mT
        semantic_loss = F.cross_entropy(
            codebook_logits.reshape(-1, codebook_logits.size(-1)),
            codebook_labels.reshape(-1),
            ignore_index=-100,
            reduction="none",
        )

        base_loss = base_loss.reshape(labels[:, 0].shape)
        semantic_loss = semantic_loss.reshape(codebook_labels.shape)

        semantic_loss_frame = semantic_loss.mean(-1)
        pad_pos = codebook_labels.sum(-1) == -100 * model.config.num_codebooks

        for loss_sample, pad in zip(semantic_loss_frame, pad_pos):
            semantic_loss_sum[~pad] += loss_sample[~pad]
            counter[~pad] += 1

        current_step += 1
        if current_step == 10:
            break

    semantic_loss = semantic_loss.cpu()
    counter = counter.cpu()
    xs, ys = [], []

    for i, (loss, count) in enumerate(zip(semantic_loss_sum, counter)):
        if count > 0:
            xs.append(i)
            ys.append((loss / count).item())  # for better loss visualization

    smoothed_ys = smooth(ys, 0.95)

    # Unload model
    del model
    torch.cuda.empty_cache()

    return xs, ys, smoothed_ys


def main():
    tokenizer = AutoTokenizer.from_pretrained("fishaudio/fish-speech-1")
    max_length = 4096

    ds = AutoAugTextDataset(
        ["data/protos/sft/云天河"],
        tokenizer=tokenizer,
        use_speaker=False,
        interactive_prob=1.0,
        max_length=max_length,
    )

    loader = DataLoader(
        ds,
        batch_size=8,
        collate_fn=TextDataCollator(tokenizer, max_length=max_length),
        num_workers=0,
        shuffle=False,
    )

    plt.figure(figsize=(10, 5), dpi=200)

    plt.xlabel("Frame")
    plt.ylabel("Loss")
    plt.yscale("log")
    plt.title("Semantic Loss")
    plt.grid(which="both", axis="both")
    plt.xlim(0, max_length)

    tests = [
        (
            "pertrain-medium",
            "dual_ar_2_codebook_medium",
            "checkpoints/text2semantic-pretrain-medium-2k-v1.pth",
        ),
        (
            "sft-medium",
            "dual_ar_2_codebook_medium",
            "checkpoints/text2semantic-sft-medium-v1.1-4k.pth",
        ),
        (
            "sft-large",
            "dual_ar_2_codebook_large",
            "checkpoints/text2semantic-sft-large-v1.1-4k.pth",
        ),
    ]

    for name, config, weight in tests:
        xs, _, smoothed_ys = analyze_one_model(loader, config, weight, max_length)
        plt.plot(xs, smoothed_ys, label=name)

    plt.legend()
    plt.savefig("semantic_loss.png")


if __name__ == "__main__":
    main()
