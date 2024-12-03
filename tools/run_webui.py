
import io
import os

import wave
from argparse import ArgumentParser
from pathlib import Path

import gradio as gr


import pyrootutils
import torch
from loguru import logger
from typing import Callable

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from fish_speech.i18n import i18n
from tools.schema import ServeTTSRequest
from tools.webui.inference_engine import inference
from tools.file import AUDIO_EXTENSIONS, list_files
from tools.llama.generate import launch_thread_safe_queue
from tools.vqgan.inference import load_model as load_decoder_model
from tools.webui.inference_engine.utils import get_inference_wrapper
from fish_speech.text.chn_text_norm.text import Text as ChnNormedText

# Make einx happy
os.environ["EINX_FILTER_TRACEBACK"] = "false"


HEADER_MD = f"""# Fish Speech

{i18n("A text-to-speech model based on VQ-GAN and Llama developed by [Fish Audio](https://fish.audio).")}  

{i18n("You can find the source code [here](https://github.com/fishaudio/fish-speech) and models [here](https://huggingface.co/fishaudio/fish-speech-1.5).")}  

{i18n("Related code and weights are released under CC BY-NC-SA 4.0 License.")}  

{i18n("We are not responsible for any misuse of the model, please consider your local laws and regulations before using it.")}  
"""

TEXTBOX_PLACEHOLDER = i18n("Put your text here.")
SPACE_IMPORTED = False


n_audios = 4

global_audio_list = []
global_error_list = []


def wav_chunk_header(sample_rate=44100, bit_depth=16, channels=1):
    buffer = io.BytesIO()

    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(bit_depth // 8)
        wav_file.setframerate(sample_rate)

    wav_header_bytes = buffer.getvalue()
    buffer.close()
    return wav_header_bytes


def normalize_text(user_input, use_normalization):
    if use_normalization:
        return ChnNormedText(raw_text=user_input).normalize()
    else:
        return user_input


def update_examples():
    examples_dir = Path("references")
    examples_dir.mkdir(parents=True, exist_ok=True)
    example_audios = list_files(examples_dir, AUDIO_EXTENSIONS, recursive=True)
    return gr.Dropdown(choices=[str(audio) for audio in example_audios] + [""])


def build_app(inference_fct : Callable):
    with gr.Blocks(theme=gr.themes.Base()) as app:
        gr.Markdown(HEADER_MD)

        # Use light theme by default
        app.load(
            None,
            None,
            js="() => {const params = new URLSearchParams(window.location.search);if (!params.has('__theme')) {params.set('__theme', '%s');window.location.search = params.toString();}}"
            % args.theme,
        )

        # Inference
        with gr.Row():
            with gr.Column(scale=3):
                text = gr.Textbox(
                    label=i18n("Input Text"), placeholder=TEXTBOX_PLACEHOLDER, lines=10
                )
                refined_text = gr.Textbox(
                    label=i18n("Realtime Transform Text"),
                    placeholder=i18n(
                        "Normalization Result Preview (Currently Only Chinese)"
                    ),
                    lines=5,
                    interactive=False,
                )

                with gr.Row():
                    normalize = gr.Checkbox(
                        label=i18n("Text Normalization"),
                        value=False,
                    )

                with gr.Row():
                    with gr.Column():
                        with gr.Tab(label=i18n("Advanced Config")):
                            with gr.Row():
                                chunk_length = gr.Slider(
                                    label=i18n("Iterative Prompt Length, 0 means off"),
                                    minimum=0,
                                    maximum=300,
                                    value=200,
                                    step=8,
                                )

                                max_new_tokens = gr.Slider(
                                    label=i18n(
                                        "Maximum tokens per batch, 0 means no limit"
                                    ),
                                    minimum=0,
                                    maximum=2048,
                                    value=0,
                                    step=8,
                                )

                            with gr.Row():
                                top_p = gr.Slider(
                                    label="Top-P",
                                    minimum=0.6,
                                    maximum=0.9,
                                    value=0.7,
                                    step=0.01,
                                )

                                repetition_penalty = gr.Slider(
                                    label=i18n("Repetition Penalty"),
                                    minimum=1,
                                    maximum=1.5,
                                    value=1.2,
                                    step=0.01,
                                )

                            with gr.Row():
                                temperature = gr.Slider(
                                    label="Temperature",
                                    minimum=0.6,
                                    maximum=0.9,
                                    value=0.7,
                                    step=0.01,
                                )
                                seed = gr.Number(
                                    label="Seed",
                                    info="0 means randomized inference, otherwise deterministic",
                                    value=0,
                                )

                        with gr.Tab(label=i18n("Reference Audio")):
                            with gr.Row():
                                gr.Markdown(
                                    i18n(
                                        "5 to 10 seconds of reference audio, useful for specifying speaker."
                                    )
                                )
                            with gr.Row():
                                reference_id = gr.Textbox(
                                    label=i18n("Reference ID"),
                                    placeholder="Leave empty to use uploaded references",
                                )

                            with gr.Row():
                                use_memory_cache = gr.Radio(
                                    label=i18n("Use Memory Cache"),
                                    choices=["never", "on-demand", "always"],
                                    value="on-demand",
                                )

                            with gr.Row():
                                reference_audio = gr.Audio(
                                    label=i18n("Reference Audio"),
                                    type="filepath",
                                )
                            with gr.Row():
                                reference_text = gr.Textbox(
                                    label=i18n("Reference Text"),
                                    lines=1,
                                    placeholder="在一无所知中，梦里的一天结束了，一个新的「轮回」便会开始。",
                                    value="",
                                )

            with gr.Column(scale=3):
                with gr.Row():
                    error = gr.HTML(
                        label=i18n("Error Message"),
                        visible=True,
                    )
                with gr.Row():
                    audio = gr.Audio(
                        label=i18n("Generated Audio"),
                        type="numpy",
                        interactive=False,
                        visible=True,
                    )

                with gr.Row():
                    with gr.Column(scale=3):
                        generate = gr.Button(
                            value="\U0001F3A7 " + i18n("Generate"), variant="primary"
                        )

        text.input(fn=normalize_text, inputs=[text, normalize], outputs=[refined_text])

        # Submit
        generate.click(
            inference_fct,
            [
                refined_text,
                normalize,
                reference_id,
                reference_audio,
                reference_text,
                max_new_tokens,
                chunk_length,
                top_p,
                repetition_penalty,
                temperature,
                seed,
                use_memory_cache,
            ],
            [audio, error],
            concurrency_limit=1,
        )

    return app


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

    inference_fct = get_inference_wrapper(
        llama_queue=llama_queue,
        decoder_model=decoder_model,
        compile=args.compile,
        precision=args.precision,
    )

    app = build_app(inference_fct)
    app.launch(show_api=True)