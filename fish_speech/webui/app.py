from pathlib import Path

import html
import io
import traceback

import gradio as gr
import librosa
import requests

from fish_speech.text import parse_text_to_segments

HEADER_MD = """
# Fish Speech

基于 VQ-GAN 和 Llama 的多语种语音合成. 感谢 Rcell 的 GPT-VITS 提供的思路.
"""

TEXTBOX_PLACEHOLDER = """在启用自动音素的情况下, 模型默认会全自动将输入文本转换为音素. 例如:
测试一下 Hugging face, BGM声音很大吗？那我改一下. 世界、こんにちは。

会被转换为:
<Segment ZH: '测试一下' -> 'c e4 sh ir4 y i2 x ia4'>
<Segment EN: ' Hugging face, BGM' -> 'HH AH1 G IH0 NG F EY1 S , B AE1 G M'>
<Segment ZH: '声音很大吗?那我改一下.' -> 'sh eng1 y in1 h en3 d a4 m a5 ? n a4 w o2 g ai3 y i2 x ia4 .'>
<Segment ZH: '世界,' -> 'sh ir4 j ie4 ,'>
<Segment JP: 'こんにちは.' -> 'k o N n i ch i w a .'>

如你所见, 最后的句子被分割为了两个部分, 因为该日文包含了汉字, 你可以使用 <jp>...</jp> 标签来指定日文优先级. 例如:
测试一下 Hugging face, BGM声音很大吗？那我改一下. <jp>世界、こんにちは。</jp>

可以看到, 日文部分被正确地分割了出来:
...
<Segment JP: '世界,こんにちは.' -> 's e k a i , k o N n i ch i w a .'>
"""


def build_html_error_message(error):
    return f"""
    <div style="color: red; font-weight: bold;">
        {html.escape(error)}
    </div>
    """


def prepare_text(
    text,
    input_mode,
    language0,
    language1,
    language2,
    enable_reference_audio,
    reference_text,
):
    lines = text.splitlines()
    languages = [language0, language1, language2]
    languages = [
        {
            "中文": "ZH",
            "日文": "JP",
            "英文": "EN",
        }[language]
        for language in languages
    ]

    if len(set(languages)) != len(languages):
        return [], build_html_error_message("语言优先级不能重复.")

    if enable_reference_audio:
        reference_text = reference_text.strip() + " "
    else:
        reference_text = ""

    if input_mode != "自动音素":
        return [
            [idx, reference_text + line, "-", "-"]
            for idx, line in enumerate(lines)
            if line.strip() != ""
        ], None

    rows = []

    for idx, line in enumerate(lines):
        if line.strip() == "":
            continue

        try:
            segments = parse_text_to_segments(reference_text + line, order=languages)
        except Exception:
            traceback.print_exc()
            err = traceback.format_exc()
            return [], build_html_error_message(f"解析 '{line}' 时发生错误. \n\n{err}")

        for segment in segments:
            rows.append([idx, segment.text, segment.language, " ".join(segment.phones)])

    return rows, None


def load_model(
    server_url,
    llama_ckpt_path,
    llama_config_name,
    tokenizer,
    vqgan_ckpt_path,
    vqgan_config_name,
    device,
    precision,
    compile_model,
):
    payload = {
        "device": device,
        "llama": {
            "config_name": llama_config_name,
            "checkpoint_path": llama_ckpt_path,
            "precision": precision,
            "tokenizer": tokenizer,
            "compile": compile_model,
        },
        "vqgan": {
            "config_name": vqgan_config_name,
            "checkpoint_path": vqgan_ckpt_path,
        },
    }

    try:
        resp = requests.put(f"{server_url}/v1/models/default", json=payload)
        resp.raise_for_status()
    except Exception:
        traceback.print_exc()
        err = traceback.format_exc()
        return build_html_error_message(f"加载模型时发生错误. \n\n{err}")

    return "模型加载成功."


def build_model_config_block():
    server_url = gr.Textbox(label="服务器地址", value="http://localhost:8000")

    with gr.Row():
        with gr.Column(scale=1):
            device = gr.Dropdown(
                label="设备",
                choices=["cpu", "cuda"],
                value="cuda",
            )
        with gr.Column(scale=1):
            precision = gr.Dropdown(
                label="精度",
                choices=["bfloat16", "float16"],
                value="float16",
            )
        with gr.Column(scale=1):
            compile_model = gr.Checkbox(
                label="编译模型",
                value=True,
            )

    llama_ckpt_path = gr.Dropdown(
        label="Llama 模型路径",
        value=str(Path("checkpoints/text2semantic-400m-v0.3-4k.pth")),
        choices=[str(pth_file) for pth_file in Path("results").rglob("*text*/*.ckpt")] + \
                     [str(pth_file) for pth_file in Path("checkpoints").rglob("*text*.pth")],
        allow_custom_value=True
    )
    llama_config_name = gr.Textbox(label="Llama 配置文件", value="text2semantic_finetune")
    tokenizer = gr.Dropdown(label="Tokenizer",
                           value="fishaudio/speech-lm-v1",
                           choices=["fishaudio/speech-lm-v1", "checkpoints"]
                           )

    vqgan_ckpt_path = gr.Dropdown(label="VQGAN 模型路径",
                                  value=str(Path("checkpoints/vqgan-v1.pth")),
                                  choices=[str(pth_file) for pth_file in Path("results").rglob("*vqgan*/*.ckpt")] + \
                                           [str(pth_file) for pth_file in Path("checkpoints").rglob("*vqgan*.pth")],
                                  allow_custom_value=True
                                  )
    vqgan_config_name = gr.Dropdown(label="VQGAN 配置文件",
                                   value="vqgan_pretrain",
                                   choices=["vqgan_pretrain", "vqgan_finetune"])

    load_model_btn = gr.Button(value="加载模型", variant="primary")
    error = gr.HTML(label="错误信息")

    load_model_btn.click(
        load_model,
        [
            server_url,
            llama_ckpt_path,
            llama_config_name,
            tokenizer,
            vqgan_ckpt_path,
            vqgan_config_name,
            device,
            precision,
            compile_model,
        ],
        [error],
    )

    return server_url


def inference(
    server_url,
    text,
    input_mode,
    language0,
    language1,
    language2,
    enable_reference_audio,
    reference_audio,
    reference_text,
    max_new_tokens,
    top_k,
    top_p,
    repetition_penalty,
    temperature,
    speaker,
):
    languages = [language0, language1, language2]
    languages = [
        {
            "中文": "zh",
            "日文": "jp",
            "英文": "en",
        }[language]
        for language in languages
    ]

    if len(set(languages)) != len(languages):
        return [], build_html_error_message("语言优先级不能重复.")

    order = ",".join(languages)
    payload = {
        "text": text,
        "prompt_text": reference_text if enable_reference_audio else None,
        "prompt_tokens": reference_audio if enable_reference_audio else None,
        "max_new_tokens": int(max_new_tokens),
        "top_k": int(top_k) if top_k > 0 else None,
        "top_p": top_p,
        "repetition_penalty": repetition_penalty,
        "temperature": temperature,
        "order": order,
        "use_g2p": input_mode == "自动音素",
        "seed": None,
        "speaker": speaker if speaker.strip() != "" else None,
    }

    try:
        resp = requests.post(f"{server_url}/v1/models/default/invoke", json=payload)
        resp.raise_for_status()
    except Exception:
        traceback.print_exc()
        err = traceback.format_exc()
        return [], build_html_error_message(f"推理时发生错误. \n\n{err}")

    content = io.BytesIO(resp.content)
    content.seek(0)
    content, sr = librosa.load(content, sr=None, mono=True)

    return (sr, content), None


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
            with gr.Tab(label="模型配置"):
                server_url = build_model_config_block()

            with gr.Tab(label="推理配置"):
                text = gr.Textbox(
                    label="输入文本", placeholder=TEXTBOX_PLACEHOLDER, lines=15
                )

                with gr.Row():
                    with gr.Tab(label="合成参数"):
                        gr.Markdown("配置常见合成参数. 自动音素会在推理时自动将文本转换为音素.")

                        input_mode = gr.Dropdown(
                            choices=["文本", "自动音素"],
                            value="文本",
                            label="输入模式",
                        )

                        max_new_tokens = gr.Slider(
                            label="最大生成 Token 数",
                            minimum=0,
                            maximum=4096,
                            value=0,  # 0 means no limit
                            step=8,
                        )

                        top_k = gr.Slider(
                            label="Top-K", minimum=0, maximum=100, value=0, step=1
                        )

                        top_p = gr.Slider(
                            label="Top-P", minimum=0, maximum=1, value=0.5, step=0.01
                        )

                        repetition_penalty = gr.Slider(
                            label="重复惩罚", minimum=0, maximum=2, value=1.5, step=0.01
                        )

                        temperature = gr.Slider(
                            label="温度", minimum=0, maximum=2, value=0.7, step=0.01
                        )

                        speaker = gr.Textbox(
                            label="说话人",
                            placeholder="说话人",
                            lines=1,
                        )

                    with gr.Tab(label="语言优先级"):
                        gr.Markdown("该参数只在自动音素转换时生效.")

                        with gr.Column(scale=1):
                            language0 = gr.Dropdown(
                                choices=["中文", "日文", "英文"],
                                label="语言 1",
                                value="中文",
                            )

                        with gr.Column(scale=1):
                            language1 = gr.Dropdown(
                                choices=["中文", "日文", "英文"],
                                label="语言 2",
                                value="日文",
                            )

                        with gr.Column(scale=1):
                            language2 = gr.Dropdown(
                                choices=["中文", "日文", "英文"],
                                label="语言 3",
                                value="英文",
                            )

                    with gr.Tab(label="参考音频"):
                        gr.Markdown("5-10 秒的参考音频, 适用于指定音色.")

                        enable_reference_audio = gr.Checkbox(
                            label="启用参考音频", value=False
                        )
                        reference_audio = gr.Audio(
                            label="参考音频",
                            value="docs/assets/audios/0_input.wav",
                            type="filepath",
                        )
                        reference_text = gr.Textbox(
                            label="参考文本",
                            placeholder="参考文本",
                            lines=1,
                            value="在一无所知中，梦里的一天结束了，一个新的「轮回」便会开始。",
                        )

                with gr.Row():
                    with gr.Column(scale=2):
                        generate = gr.Button(value="合成", variant="primary")
                    with gr.Column(scale=1):
                        clear = gr.Button(value="清空")

        with gr.Column(scale=3):
            error = gr.HTML(label="错误信息")
            parsed_text = gr.Dataframe(
                label="解析结果 (仅参考)", headers=["ID", "文本", "语言", "音素"]
            )
            audio = gr.Audio(label="合成音频", type="numpy")

    # Language & Text Parsing
    kwargs = dict(
        inputs=[
            text,
            input_mode,
            language0,
            language1,
            language2,
            enable_reference_audio,
            reference_text,
        ],
        outputs=[parsed_text, error],
        trigger_mode="always_last",
    )
    text.change(prepare_text, **kwargs)
    input_mode.change(prepare_text, **kwargs)
    language0.change(prepare_text, **kwargs)
    language1.change(prepare_text, **kwargs)
    language2.change(prepare_text, **kwargs)
    enable_reference_audio.change(prepare_text, **kwargs)

    # Submit
    generate.click(
        inference,
        [
            server_url,
            text,
            input_mode,
            language0,
            language1,
            language2,
            enable_reference_audio,
            reference_audio,
            reference_text,
            max_new_tokens,
            top_k,
            top_p,
            repetition_penalty,
            temperature,
            speaker,
        ],
        [audio, error],
    )


if __name__ == "__main__":
    app.launch(show_api=False)
