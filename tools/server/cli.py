"""API command-line options, independently usable without the ML stack."""

from argparse import ArgumentParser


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--model", help="Hugging Face model ID (default: fishaudio/s2-pro)"
    )
    parser.add_argument("--revision", help="Hub model branch, tag, or commit")
    parser.add_argument("--offline", action="store_true", help="Use cached files only")
    parser.add_argument(
        "--codec-model",
        default="fishaudio/s2-pro",
        help="Hub fallback codec repository",
    )
    parser.add_argument(
        "--codec-revision", help="Separate revision for the fallback codec"
    )
    parser.add_argument("--mode", type=str, choices=["tts"], default="tts")
    parser.add_argument(
        "--llama-checkpoint-path",
        type=str,
        default=None,
        help="S2-Pro (dual_ar) or Drama 3 (fish_omni) checkpoint directory; "
        "the model type is detected from its config.json.",
    )
    parser.add_argument(
        "--decoder-checkpoint-path",
        type=str,
        default=None,
        help="DAC codec weights. Defaults to <checkpoint>/codec.pth, else the "
        "sibling s2-pro/codec.pth.",
    )
    parser.add_argument("--decoder-config-name", type=str, default="modded_dac_vq")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Drama 3 only: plain 'cuda' takes the first min(4, visible) GPUs, or list "
        "them ('cuda:0,cuda:1,...'); the last one holds the tower and the codec.",
    )
    parser.add_argument(
        "--half",
        action="store_true",
        help="float16 instead of bfloat16 (Drama 3 switches by itself before SM80).",
    )
    parser.add_argument(
        "--compile", action="store_true", help="torch.compile (S2-Pro only)."
    )
    parser.add_argument("--max-text-length", type=int, default=0)
    parser.add_argument("--listen", type=str, default="127.0.0.1:8080")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--api-key", type=str, default=None)

    args = parser.parse_args()
    if args.model and args.llama_checkpoint_path:
        parser.error("--model and --llama-checkpoint-path are mutually exclusive")
    return args
