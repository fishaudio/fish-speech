import gc
import html
import io
import os
import queue
import wave
from argparse import ArgumentParser
from functools import partial, wraps
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
from tools.llama.generate import launch_thread_safe_queue
from tools.vqgan.inference import load_model as load_vqgan_model

# Make einx happy
os.environ["EINX_FILTER_TRACEBACK"] = "false"


HEADER_MD = f"""# Fish Speech

{i18n("A text-to-speech model based on VQ-GAN and Llama developed by [Fish Audio](https://fish.audio).")}  

{i18n("You can find the source code [here](https://github.com/fishaudio/fish-speech) and models [here](https://huggingface.co/fishaudio/fish-speech-1).")}  

{i18n("Related code are released under BSD-3-Clause License, and weights are released under CC BY-NC-SA 4.0 License.")}  

{i18n("We are not responsible for any misuse of the model, please consider your local laws and regulations before using it.")}  
"""

TEXTBOX_PLACEHOLDER = i18n("Put your text here.")
SPACE_IMPORTED = False

reference_wavs = ["请选择参考音频,或者自己上传"]

for name in os.listdir("./参考音频/"):
    reference_wavs.append(name)


def change_choices():

    reference_wavs = ["请选择参考音频,或者自己上传"]

    for name in os.listdir("./参考音频/"):
        reference_wavs.append(name)
    
    return {"choices":reference_wavs, "__type__": "update"}


def change_wav(audio_path):

    text = audio_path.replace(".wav","").replace(".mp3","")

    return f"./参考音频/{audio_path}",text


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
    speaker,
    streaming=False,
):
    if args.max_gradio_length > 0 and len(text) > args.max_gradio_length:
        return (
            None,
            i18n("Text is too long, please keep it under {} characters.").format(
                args.max_gradio_length
            ),
        )

    # Parse reference audio aka prompt
    prompt_tokens = None
    if enable_reference_audio and reference_audio is not None:
        # reference_audio_sr, reference_audio_content = reference_audio
        reference_audio_content, _ = librosa.load(
            reference_audio, sr=vqgan_model.sampling_rate, mono=True
        )
        audios = torch.from_numpy(reference_audio_content).to(vqgan_model.device)[
            None, None, :
        ]

        logger.info(
            f"Loaded audio with {audios.shape[2] / vqgan_model.sampling_rate:.2f} seconds"
        )

        # VQ Encoder
        audio_lengths = torch.tensor(
            [audios.shape[2]], device=vqgan_model.device, dtype=torch.long
        )
        prompt_tokens = vqgan_model.encode(audios, audio_lengths)[0][0]

    # LLAMA Inference
    request = dict(
        tokenizer=llama_tokenizer,
        device=vqgan_model.device,
        max_new_tokens=max_new_tokens,
        text=text,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        temperature=temperature,
        compile=args.compile,
        iterative_prompt=chunk_length > 0,
        chunk_length=chunk_length,
        max_length=args.max_length,
        speaker=speaker if speaker else None,
        prompt_tokens=prompt_tokens if enable_reference_audio else None,
        prompt_text=reference_text if enable_reference_audio else None,
        is_streaming=True,  # Always streaming
    )

    payload = dict(
        response_queue=queue.Queue(),
        request=request,
    )
    llama_queue.put(payload)

    if streaming:
        yield wav_chunk_header(), None

    segments = []
    while True:
        result = payload["response_queue"].get()
        if result == "next":
            # TODO: handle next sentence
            continue

        if result == "done":
            if payload["success"] is False:
                yield None, build_html_error_message(payload["response"])
            break

        # VQGAN Inference
        feature_lengths = torch.tensor([result.shape[1]], device=vqgan_model.device)
        fake_audios = vqgan_model.decode(
            indices=result[None], feature_lengths=feature_lengths, return_audios=True
        )[0, 0]
        fake_audios = fake_audios.float().cpu().numpy()
        fake_audios = np.concatenate([fake_audios, np.zeros((11025,))], axis=0)

        if streaming:
            yield (fake_audios * 32768).astype(np.int16).tobytes(), None
        else:
            segments.append(fake_audios)

    if streaming is False:
        audio = np.concatenate(segments, axis=0)
        yield (vqgan_model.sampling_rate, audio), None

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()


inference_stream = partial(inference, streaming=True)


def wav_chunk_header(sample_rate=44100, bit_depth=16, channels=1):
    buffer = io.BytesIO()

    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(bit_depth // 8)
        wav_file.setframerate(sample_rate)

    wav_header_bytes = buffer.getvalue()
    buffer.close()
    return wav_header_bytes


def build_app():
    with gr.Blocks(theme=gr.themes.Base()) as app:
        gr.Markdown(HEADER_MD)

        # Use light theme by default
        app.load(
            None,
            None,
            js="() => {const params = new URLSearchParams(window.location.search);if (!params.has('__theme')) {params.set('__theme', 'light');window.location.search = params.toString();}}",
        )

        # Inference
        with gr.Row():
            with gr.Column(scale=3):
                text = gr.Textbox(
                    label=i18n("Input Text"), placeholder=TEXTBOX_PLACEHOLDER, lines=15
                )

                with gr.Row():
                    with gr.Tab(label=i18n("Advanced Config")):
                        chunk_length = gr.Slider(
                            label=i18n("Iterative Prompt Length, 0 means off"),
                            minimum=0,
                            maximum=500,
                            value=30,
                            step=8,
                        )

                        max_new_tokens = gr.Slider(
                            label=i18n("Maximum tokens per batch, 0 means no limit"),
                            minimum=0,
                            maximum=args.max_length,
                            value=0,  # 0 means no limit
                            step=8,
                        )

                        top_p = gr.Slider(
                            label="Top-P", minimum=0, maximum=1, value=0.7, step=0.01
                        )

                        repetition_penalty = gr.Slider(
                            label=i18n("Repetition Penalty"),
                            minimum=0,
                            maximum=2,
                            value=1.5,
                            step=0.01,
                        )

                        temperature = gr.Slider(
                            label="Temperature",
                            minimum=0,
                            maximum=2,
                            value=0.7,
                            step=0.01,
                        )

                        speaker = gr.Textbox(
                            label=i18n("Speaker"),
                            placeholder=i18n("Type name of the speaker"),
                            lines=1,
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

                        wavs_dropdown = gr.Dropdown(label="参考音频列表",choices=reference_wavs,value="请选择参考音频,或者自己上传",interactive=True)
                        refresh_button = gr.Button("刷新参考音频音频列表")
                        refresh_button.click(fn=change_choices, inputs=[], outputs=[wavs_dropdown])
                        
                        reference_audio = gr.Audio(
                            label=i18n("Reference Audio"),
                            type="filepath",
                        )
                        reference_text = gr.Textbox(
                            label=i18n("Reference Text"),
                            placeholder=i18n("Reference Text"),
                            lines=1,
                            value="在一无所知中，梦里的一天结束了，一个新的「轮回」便会开始。",
                        )

                        wavs_dropdown.change(change_wav,[wavs_dropdown],[reference_audio,reference_text])

            with gr.Column(scale=3):
                with gr.Row():
                    error = gr.HTML(label=i18n("Error Message"))
                with gr.Row():
                    audio = gr.Audio(
                        label=i18n("Generated Audio"),
                        type="numpy",
                        interactive=False,
                    )
                with gr.Row():
                    stream_audio = gr.Audio(
                        label=i18n("Streaming Audio"),
                        streaming=True,
                        autoplay=True,
                        interactive=False,
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
        # # Submit
        generate.click(
            inference,
            [
                text,
                enable_reference_audio,
                reference_audio,
                reference_text,
                max_new_tokens,
                chunk_length,
                top_p,
                repetition_penalty,
                temperature,
                speaker,
            ],
            [audio, error],
            concurrency_limit=1,
        )
        generate_stream.click(
            inference_stream,
            [
                text,
                enable_reference_audio,
                reference_audio,
                reference_text,
                max_new_tokens,
                chunk_length,
                top_p,
                repetition_penalty,
                temperature,
                speaker,
            ],
            [stream_audio, error],
            concurrency_limit=10,
        )
    return app


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--llama-checkpoint-path",
        type=Path,
        default="checkpoints/text2semantic-sft-medium-v1-4k.pth",
    )
    parser.add_argument(
        "--llama-config-name", type=str, default="dual_ar_2_codebook_large"
    )
    parser.add_argument(
        "--vqgan-checkpoint-path",
        type=Path,
        default="checkpoints/vq-gan-group-fsq-2x1024.pth",
    )
    parser.add_argument("--vqgan-config-name", type=str, default="vqgan_pretrain")
    parser.add_argument("--tokenizer", type=str, default="fishaudio/fish-speech-1")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--half", action="store_true")
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--max-gradio-length", type=int, default=0)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    args.precision = torch.half if args.half else torch.bfloat16

    logger.info("Loading Llama model...")
    llama_queue = launch_thread_safe_queue(
        config_name=args.llama_config_name,
        checkpoint_path=args.llama_checkpoint_path,
        device=args.device,
        precision=args.precision,
        max_length=args.max_length,
        compile=args.compile,
    )
    llama_tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    logger.info("Llama model loaded, loading VQ-GAN model...")

    vqgan_model = load_vqgan_model(
        config_name=args.vqgan_config_name,
        checkpoint_path=args.vqgan_checkpoint_path,
        device=args.device,
    )

    logger.info("VQ-GAN model loaded, warming up...")

    # Dry run to check if the model is loaded correctly and avoid the first-time latency
    list(
        inference(
            text="Hello, world!",
            enable_reference_audio=False,
            reference_audio=None,
            reference_text="",
            max_new_tokens=0,
            chunk_length=0,
            top_p=0.7,
            repetition_penalty=1.5,
            temperature=0.7,
            speaker=None,
        )
    )

    logger.info("Warming up done, launching the web UI...")

    app = build_app()
    app.launch(show_api=True)
