import html
import traceback

import gradio as gr

from fish_speech.text import parse_text_to_segments, segments_to_phones

HEADER_MD = """
# Fish Speech

基于 VITS 和 GPT 的多语种语音合成. 项目很大程度上基于 Rcell 的 GPT-VITS.
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


def prepare_text(text, input_mode, language0, language1, language2):
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

    if input_mode != "自动音素转换":
        return [
            [idx, line, "-", "-"]
            for idx, line in enumerate(lines)
            if line.strip() != ""
        ], None

    rows = []

    for idx, line in enumerate(lines):
        if line.strip() == "":
            continue

        try:
            segments = parse_text_to_segments(line, order=languages)
        except Exception:
            traceback.print_exc()
            err = traceback.format_exc()
            return [], build_html_error_message(f"解析 '{line}' 时发生错误. \n\n{err}")

        for segment in segments:
            rows.append([idx, segment.text, segment.language, segment.phones])

    return rows, None


with gr.Blocks(theme=gr.themes.Base()) as app:
    gr.Markdown(HEADER_MD)

    with gr.Row():
        with gr.Column(scale=3):
            text = gr.Textbox(label="输入文本", placeholder=TEXTBOX_PLACEHOLDER, lines=3)

            with gr.Row():
                with gr.Tab(label="合成参数"):
                    gr.Markdown("配置常见合成参数.")

                    input_mode = gr.Dropdown(
                        choices=["手动输入音素/文本", "自动音素转换"],
                        value="手动输入音素/文本",
                        label="输入模式",
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

            with gr.Row():
                with gr.Column(scale=2):
                    generate = gr.Button(value="合成", variant="primary")
                with gr.Column(scale=1):
                    clear = gr.Button(value="清空")

        with gr.Column(scale=3):
            error = gr.HTML(label="错误信息")
            parsed_text = gr.Dataframe(label="解析结果", headers=["ID", "文本", "语言", "音素"])
            audio = gr.Audio(label="合成音频")

    # Language & Text Parsing
    kwargs = dict(
        inputs=[text, input_mode, language0, language1, language2],
        outputs=[parsed_text, error],
        trigger_mode="always_last",
    )
    text.change(prepare_text, **kwargs)
    input_mode.change(prepare_text, **kwargs)
    language0.change(prepare_text, **kwargs)
    language1.change(prepare_text, **kwargs)
    language2.change(prepare_text, **kwargs)

if __name__ == "__main__":
    app.launch(show_api=False)
