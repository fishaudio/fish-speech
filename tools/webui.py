import gc
import html
import io
import os
import queue
import wave
from argparse import ArgumentParser
from functools import partial
from pathlib import Path

import gradio as gr
import librosa
import numpy as np
import pyrootutils
import torch
from loguru import logger
from transformers import AutoTokenizer

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)


from fish_speech.i18n import i18n
from fish_speech.text.chn_text_norm.text import Text as ChnNormedText
from fish_speech.utils import autocast_exclude_mps
from tools.api import decode_vq_tokens, encode_reference
from tools.llama.generate import (
    GenerateRequest,
    GenerateResponse,
    WrappedGenerateResponse,
    launch_thread_safe_queue,
)
from tools.vqgan.inference import load_model as load_decoder_model

# Make einx happy
os.environ["EINX_FILTER_TRACEBACK"] = "false"


HEADER_MD = f"""# Fish Speech

{i18n("A text-to-speech model based on VQ-GAN and Llama developed by [Fish Audio](https://fish.audio).")}  

{i18n("You can find the source code [here](https://github.com/fishaudio/fish-speech) and models [here](https://huggingface.co/fishaudio/fish-speech-1.4).")}  

{i18n("Related code and weights are released under CC BY-NC-SA 4.0 License.")}  

{i18n("We are not responsible for any misuse of the model, please consider your local laws and regulations before using it.")}  
"""

TEXTBOX_PLACEHOLDER = i18n("Put your text here.")
SPACE_IMPORTED = False


def build_html_error_message(error):
    return f"""
    <div style="color: red; 
    font-weight: bold;">
        {html.escape(str(error))}
    </div>
    """


@torch.inference_mode()
def inference(
    text,
    enable_reference_audio,
    reference_audio,
    reference_text,
    max_new_tokens,
    chunk_length,
    top_p,
    repetition_penalty,
    temperature,
    streaming=False,
):
    if args.max_gradio_length > 0 and len(text) > args.max_gradio_length:
        return (
            None,
            None,
            i18n("Text is too long, please keep it under {} characters.").format(
                args.max_gradio_length
            ),
        )

    # Parse reference audio aka prompt
    prompt_tokens = encode_reference(
        decoder_model=decoder_model,
        reference_audio=reference_audio,
        enable_reference_audio=enable_reference_audio,
    )

    # LLAMA Inference
    request = dict(
        device=decoder_model.device,
        max_new_tokens=max_new_tokens,
        text=text,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        temperature=temperature,
        compile=args.compile,
        iterative_prompt=chunk_length > 0,
        chunk_length=chunk_length,
        max_length=2048,
        prompt_tokens=prompt_tokens if enable_reference_audio else None,
        prompt_text=reference_text if enable_reference_audio else None,
    )

    response_queue = queue.Queue()
    llama_queue.put(
        GenerateRequest(
            request=request,
            response_queue=response_queue,
        )
    )

    if streaming:
        yield wav_chunk_header(), None, None

    segments = []

    while True:
        result: WrappedGenerateResponse = response_queue.get()
        if result.status == "error":
            yield None, None, build_html_error_message(result.response)
            break

        result: GenerateResponse = result.response
        if result.action == "next":
            break

        with autocast_exclude_mps(
            device_type=decoder_model.device.type, dtype=args.precision
        ):
            fake_audios = decode_vq_tokens(
                decoder_model=decoder_model,
                codes=result.codes,
            )

        fake_audios = fake_audios.float().cpu().numpy()
        segments.append(fake_audios)

        if streaming:
            yield (fake_audios * 32768).astype(np.int16).tobytes(), None, None

    if len(segments) == 0:
        return (
            None,
            None,
            build_html_error_message(
                i18n("No audio generated, please check the input text.")
            ),
        )

    # No matter streaming or not, we need to return the final audio
    audio = np.concatenate(segments, axis=0)
    yield None, (decoder_model.spec_transform.sample_rate, audio), None

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()


inference_stream = partial(inference, streaming=True)

n_audios = 4

global_audio_list = []
global_error_list = []


def inference_wrapper(
    text,
    enable_reference_audio,
    reference_audio,
    reference_text,
    max_new_tokens,
    chunk_length,
    top_p,
    repetition_penalty,
    temperature,
    batch_infer_num,
):
    audios = []
    errors = []

    for _ in range(batch_infer_num):
        result = inference(
            text,
            enable_reference_audio,
            reference_audio,
            reference_text,
            max_new_tokens,
            chunk_length,
            top_p,
            repetition_penalty,
            temperature,
        )

        _, audio_data, error_message = next(result)

        audios.append(
            gr.Audio(value=audio_data if audio_data else None, visible=True),
        )
        errors.append(
            gr.HTML(value=error_message if error_message else None, visible=True),
        )

    for _ in range(batch_infer_num, n_audios):
        audios.append(
            gr.Audio(value=None, visible=False),
        )
        errors.append(
            gr.HTML(value=None, visible=False),
        )

    return None, *audios, *errors


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


asr_model = None


def build_app():
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
                    if_refine_text = gr.Checkbox(
                        label=i18n("Text Normalization"),
                        value=False,
                        scale=1,
                    )

                with gr.Row():
                    with gr.Tab(label=i18n("Advanced Config")):
                        chunk_length = gr.Slider(
                            label=i18n("Iterative Prompt Length, 0 means off"),
                            minimum=50,
                            maximum=300,
                            value=200,
                            step=8,
                        )

                        max_new_tokens = gr.Slider(
                            label=i18n("Maximum tokens per batch, 0 means no limit"),
                            minimum=0,
                            maximum=2048,
                            value=1024,  # 0 means no limit
                            step=8,
                        )

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

                        temperature = gr.Slider(
                            label="Temperature",
                            minimum=0.6,
                            maximum=0.9,
                            value=0.7,
                            step=0.01,
                        )

                    with gr.Tab(label=i18n("Reference Audio")):
                        gr.Markdown(
                            i18n(
                                "5 to 10 seconds of reference audio, useful for specifying speaker."
                            )
                        )

                        enable_reference_audio = gr.Checkbox(
                            label=i18n("Enable Reference Audio"),
                        )

                        # Add dropdown for selecting example audio files
                        examples_dir = Path("examples")
                        if not examples_dir.exists():
                            examples_dir.mkdir()
                        example_audio_files = [
                            f.name for f in examples_dir.glob("*.wav")
                        ] + [f.name for f in examples_dir.glob("*.mp3")]
                        example_audio_dropdown = gr.Dropdown(
                            label=i18n("Select Example Audio"),
                            choices=[""] + example_audio_files,
                            value="",
                        )

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
                    with gr.Tab(label=i18n("Batch Inference")):
                        batch_infer_num = gr.Slider(
                            label="Batch infer nums",
                            minimum=1,
                            maximum=n_audios,
                            step=1,
                            value=1,
                        )

            with gr.Column(scale=3):
                for _ in range(n_audios):
                    with gr.Row():
                        error = gr.HTML(
                            label=i18n("Error Message"),
                            visible=True if _ == 0 else False,
                        )
                        global_error_list.append(error)
                    with gr.Row():
                        audio = gr.Audio(
                            label=i18n("Generated Audio"),
                            type="numpy",
                            interactive=False,
                            visible=True if _ == 0 else False,
                        )
                        global_audio_list.append(audio)

                with gr.Row():
                    stream_audio = gr.Audio(
                        label=i18n("Streaming Audio"),
                        streaming=True,
                        autoplay=True,
                        interactive=False,
                        show_download_button=True,
                    )
                with gr.Row():
                    with gr.Column(scale=3):
                        generate = gr.Button(
                            value="\U0001F3A7 " + i18n("Generate"), variant="primary"
                        )
                        generate_stream = gr.Button(
                            value="\U0001F3A7 " + i18n("Streaming Generate"),
                            variant="primary",
                        )

        text.input(
            fn=normalize_text, inputs=[text, if_refine_text], outputs=[refined_text]
        )

        def select_example_audio(audio_file):
            if audio_file:
                audio_path = examples_dir / audio_file
                lab_file = audio_path.with_suffix(".lab")

                if lab_file.exists():
                    lab_content = lab_file.read_text(encoding="utf-8").strip()
                else:
                    lab_content = ""

                return str(audio_path), lab_content, True
            return None, "", False

        # Connect the dropdown to update reference audio and text
        example_audio_dropdown.change(
            fn=select_example_audio,
            inputs=[example_audio_dropdown],
            outputs=[reference_audio, reference_text, enable_reference_audio],
        )

        # # Submit
        generate.click(
            inference_wrapper,
            [
                refined_text,
                enable_reference_audio,
                reference_audio,
                reference_text,
                max_new_tokens,
                chunk_length,
                top_p,
                repetition_penalty,
                temperature,
                batch_infer_num,
            ],
            [stream_audio, *global_audio_list, *global_error_list],
            concurrency_limit=1,
        )

        generate_stream.click(
            inference_stream,
            [
                refined_text,
                enable_reference_audio,
                reference_audio,
                reference_text,
                max_new_tokens,
                chunk_length,
                top_p,
                repetition_penalty,
                temperature,
            ],
            [stream_audio, global_audio_list[0], global_error_list[0]],
            concurrency_limit=10,
        )
    return app


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--llama-checkpoint-path",
        type=Path,
        default="checkpoints/fish-speech-1.4",
    )
    parser.add_argument(
        "--decoder-checkpoint-path",
        type=Path,
        default="checkpoints/fish-speech-1.4/firefly-gan-vq-fsq-8x1024-21hz-generator.pth",
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
            text="Hello, world!",
            enable_reference_audio=False,
            reference_audio=None,
            reference_text="",
            max_new_tokens=1024,
            chunk_length=200,
            top_p=0.7,
            repetition_penalty=1.2,
            temperature=0.7,
        )
    )

    logger.info("Warming up done, launching the web UI...")

    app = build_app()
    app.launch(show_api=True)
