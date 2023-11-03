import gradio as gr

HEADER_MD = """
# Fish Speech

基于 VITS 和 GPT 的多语种语音合成. 项目很大程度上基于 Rcell 的 GPT-VITS.
"""

with gr.Blocks(theme=gr.themes.Base()) as app:
    gr.Markdown(HEADER_MD)

    with gr.Row():
        with gr.Column(scale=5):
            text = gr.Textbox(lines=5, label="输入文本")

            with gr.Row():
                with gr.Tab(label="合成参数"):
                    gr.Markdown("配置常见的合成参数.")

                    input_mode = gr.Dropdown(
                        choices=["手动输入音素/文本", "自动音素转换"],
                        value="手动输入音素/文本",
                        label="输入模式",
                    )

                with gr.Tab(label="语言优先级"):
                    gr.Markdown("该参数只在自动音素转换时生效.")

                    with gr.Column(scale=1):
                        language0 = gr.Dropdown(
                            choices=["中文", "日文", "英文", "无"],
                            label="语言 1",
                            value="中文",
                        )

                    with gr.Column(scale=1):
                        language1 = gr.Dropdown(
                            choices=["中文", "日文", "英文", "无"],
                            label="语言 2",
                            value="英文",
                        )

                    with gr.Column(scale=1):
                        language2 = gr.Dropdown(
                            choices=["中文", "日文", "英文", "无"],
                            label="语言 3",
                            value="无",
                        )

            with gr.Row():
                with gr.Column(scale=2):
                    generate = gr.Button(value="合成", variant="primary")
                with gr.Column(scale=1):
                    clear = gr.Button(value="清空")

        with gr.Column(scale=3):
            audio = gr.Audio(label="合成音频")

    generate.click(lambda: None, [input_mode], [audio])
    # dark_mode.link(lambda val: app.set_theme(gr.themes.Dark() if val else gr.themes.Default()))

if __name__ == "__main__":
    app.launch(show_api=False)
