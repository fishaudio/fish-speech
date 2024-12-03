import os
import torch
import pyrootutils
from pathlib import Path
from loguru import logger
from argparse import ArgumentParser

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from tools.schema import ServeTTSRequest
from tools.webui.web_app import build_app
from tools.webui.inference_engine import inference
from tools.llama.generate import launch_thread_safe_queue
from tools.vqgan.inference import load_model as load_decoder_model
from tools.webui.inference_engine.utils import get_inference_wrapper

# Make einx happy
os.environ["EINX_FILTER_TRACEBACK"] = "false"


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--llama-checkpoint-path",
        type=Path,
        default="checkpoints/fish-speech-1.5",
    )
    parser.add_argument(
        "--decoder-checkpoint-path",
        type=Path,
        default="checkpoints/fish-speech-1.5/firefly-gan-vq-fsq-8x1024-21hz-generator.pth",
    )
    parser.add_argument("--decoder-config-name", type=str, default="firefly_gan_vq")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--half", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--max-gradio-length", type=int, default=0)
    parser.add_argument("--theme", type=str, default="light")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    args.precision = torch.half if args.half else torch.bfloat16

    # Check if CUDA is available
    if not torch.cuda.is_available():
        logger.info("CUDA is not available, running on CPU.")
        args.device = "cpu"

    logger.info("Loading Llama model...")
    llama_queue = launch_thread_safe_queue(
        checkpoint_path=args.llama_checkpoint_path,
        device=args.device,
        precision=args.precision,
        compile=args.compile,
    )

    logger.info("Llama model loaded, loading VQ-GAN model...")
    decoder_model = load_decoder_model(
        config_name=args.decoder_config_name,
        checkpoint_path=args.decoder_checkpoint_path,
        device=args.device,
    )

    logger.info("Decoder model loaded, warming up...")

    # Dry run to check if the model is loaded correctly and avoid the first-time latency
    list(
        inference(
            ServeTTSRequest(
                text="Hello world.",
                references=[],
                reference_id=None,
                max_new_tokens=0,
                chunk_length=200,
                top_p=0.7,
                repetition_penalty=1.5,
                temperature=0.7,
                format="wav",
                decoder_model=decoder_model,
                llama_queue=llama_queue,
                compile=args.compile,
                precision=args.precision,
            )
        )
    )

    logger.info("Warming up done, launching the web UI...")

    # Get the inference function with the immutable arguments
    inference_fct = get_inference_wrapper(
        llama_queue=llama_queue,
        decoder_model=decoder_model,
        compile=args.compile,
        precision=args.precision,
    )

    app = build_app(inference_fct, args.theme)
    app.launch(show_api=True)