import os
import subprocess
import webbrowser

import gradio as gr
import librosa


def gen_prompt(audio_path, vq_gan_path):
    subprocess.run(
        f'python "tools/vqgan/inference.py" -i "{audio_path}" --checkpoint-path "{vq_gan_path}"',
        shell=True,
    )
    return "Success"


def gen_token(text, text_prompt, llama_path, sample_num, spk, no_g2p):
    command = f'python tools/llama/generate.py --text "{text}" --prompt-text "{text_prompt}" --prompt-tokens fake.npy --checkpoint-path "{llama_path}" --num-samples {sample_num} --speaker "{spk}"'
    if no_g2p:
        command += " --no-g2p"
    subprocess.run(command, shell=True)
    return {
        "__type__": "update",
        "choices": [f"codes_{i}.npy" for i in range(sample_num)],
    }, "Success"


def gen_audio(sample_id, vq_gan_path):
    subprocess.run(
        f'python "tools/vqgan/inference.py" -i "{sample_id}" --checkpoint-path "{vq_gan_path}"',
        shell=True,
    )
    audio, sr = librosa.load(f"fake.wav", sr=22050)
    return (sr, audio), "Success"


if __name__ == "__main__":
    vq_gan_models = ["checkpoints/vqgan-v1.pth"]
    llama_models = ["checkpoints/text2semantic-400m-v0.2-4k.pth"]

    ft_vq_gan_ckpt_path = "results/vqgan/checkpoints"
    if os.path.exists(ft_vq_gan_ckpt_path):
        vq_gan_models += [
            os.path.join(ft_vq_gan_ckpt_path, i)
            for i in os.listdir(ft_vq_gan_ckpt_path)
        ]

    ft_llama_ckpt_path = "results/text2semantic_400m_finetune_spk/checkpoints"
    if os.path.exists(ft_llama_ckpt_path):
        llama_models += [
            os.path.join(ft_llama_ckpt_path, i) for i in os.listdir(ft_llama_ckpt_path)
        ]

    with gr.Blocks() as app:
        with gr.Row():
            with gr.Column():
                vq_gan_model = gr.Dropdown(label="VQ-GAN model", choices=vq_gan_models)
                llama_model = gr.Dropdown(label="LLaMA model", choices=llama_models)
                text = gr.TextArea(label="Text to generate")
                spk = gr.Textbox(label="Speaker name")
                audio_prompt = gr.Audio(label="Reference audio", type="filepath")
                text_prompt = gr.Textbox(
                    label="Text corresponding to the reference audio"
                )
                sample_num = gr.Slider(
                    label="Generated token pairs",
                    value=2,
                    minimum=1,
                    maximum=10,
                    step=1,
                )
                no_g2p = gr.Checkbox(label="Don't use G2P", value=False)
            with gr.Column():
                gen_audio_token = gr.Button("1. Generate reference audio's tokens")
                gen_text_token = gr.Button("2. Generate output tokens")
                sample_id = gr.Dropdown(label="Choose token pair", choices=[])
                decode_audio = gr.Button("3. Decode tokens")
                text_output = gr.Textbox(label="Info")
                audio_output = gr.Audio(label="Output")

        gen_audio_token.click(
            gen_prompt, inputs=[audio_prompt, vq_gan_model], outputs=[text_output]
        )
        gen_text_token.click(
            gen_token,
            inputs=[text, text_prompt, llama_model, sample_num, spk, no_g2p],
            outputs=[sample_id, text_output],
        )
        decode_audio.click(
            gen_audio,
            inputs=[sample_id, vq_gan_model],
            outputs=[audio_output, text_output],
        )

    app.launch(share=False, server_port=7860)
    webbrowser.open("http://127.0.0.1:7860")
