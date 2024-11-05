import io
import re
import wave

import gradio as gr
import numpy as np

from .fish_e2e import FishE2EAgent, FishE2EEventType
from .schema import ServeMessage, ServeTextPart, ServeVQPart


def wav_chunk_header(sample_rate=44100, bit_depth=16, channels=1):
    buffer = io.BytesIO()

    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(bit_depth // 8)
        wav_file.setframerate(sample_rate)

    wav_header_bytes = buffer.getvalue()
    buffer.close()
    return wav_header_bytes


class ChatState:
    def __init__(self):
        self.conversation = []
        self.added_systext = False
        self.added_sysaudio = False

    def get_history(self):
        results = []
        for msg in self.conversation:
            results.append({"role": msg.role, "content": self.repr_message(msg)})

        # Process assistant messages to extract questions and update user messages
        for i, msg in enumerate(results):
            if msg["role"] == "assistant":
                match = re.search(r"Question: (.*?)\n\nResponse:", msg["content"])
                if match and i > 0 and results[i - 1]["role"] == "user":
                    # Update previous user message with extracted question
                    results[i - 1]["content"] += "\n" + match.group(1)
                    # Remove the Question/Answer format from assistant message
                    msg["content"] = msg["content"].split("\n\nResponse: ", 1)[1]
        return results

    def repr_message(self, msg: ServeMessage):
        response = ""
        for part in msg.parts:
            if isinstance(part, ServeTextPart):
                response += part.text
            elif isinstance(part, ServeVQPart):
                response += f"<audio {len(part.codes[0]) / 21:.2f}s>"
        return response


def clear_fn():
    return [], ChatState(), None, None, None


async def process_audio_input(
    sys_audio_input, sys_text_input, audio_input, state: ChatState, text_input: str
):
    if audio_input is None and not text_input:
        raise gr.Error("No input provided")

    agent = FishE2EAgent()  # Create new agent instance for each request

    # Convert audio input to numpy array
    if isinstance(audio_input, tuple):
        sr, audio_data = audio_input
    elif text_input:
        sr = 44100
        audio_data = None
    else:
        raise gr.Error("Invalid audio format")

    if isinstance(sys_audio_input, tuple):
        sr, sys_audio_data = sys_audio_input
    else:
        sr = 44100
        sys_audio_data = None

    def append_to_chat_ctx(
        part: ServeTextPart | ServeVQPart, role: str = "assistant"
    ) -> None:
        if not state.conversation or state.conversation[-1].role != role:
            state.conversation.append(ServeMessage(role=role, parts=[part]))
        else:
            state.conversation[-1].parts.append(part)

    if state.added_systext is False and sys_text_input:
        state.added_systext = True
        append_to_chat_ctx(ServeTextPart(text=sys_text_input), role="system")
    if text_input:
        append_to_chat_ctx(ServeTextPart(text=text_input), role="user")
        audio_data = None

    result_audio = b""
    async for event in agent.stream(
        sys_audio_data,
        audio_data,
        sr,
        1,
        chat_ctx={
            "messages": state.conversation,
            "added_sysaudio": state.added_sysaudio,
        },
    ):
        if event.type == FishE2EEventType.USER_CODES:
            append_to_chat_ctx(ServeVQPart(codes=event.vq_codes), role="user")
        elif event.type == FishE2EEventType.SPEECH_SEGMENT:
            append_to_chat_ctx(ServeVQPart(codes=event.vq_codes))
            yield state.get_history(), wav_chunk_header() + event.frame.data, None, None
        elif event.type == FishE2EEventType.TEXT_SEGMENT:
            append_to_chat_ctx(ServeTextPart(text=event.text))
            yield state.get_history(), None, None, None

    yield state.get_history(), None, None, None


async def process_text_input(
    sys_audio_input, sys_text_input, state: ChatState, text_input: str
):
    async for event in process_audio_input(
        sys_audio_input, sys_text_input, None, state, text_input
    ):
        yield event


def create_demo():
    with gr.Blocks() as demo:
        state = gr.State(ChatState())

        with gr.Row():
            # Left column (70%) for chatbot and notes
            with gr.Column(scale=7):
                chatbot = gr.Chatbot(
                    [],
                    elem_id="chatbot",
                    bubble_full_width=False,
                    height=600,
                    type="messages",
                )

                # notes = gr.Markdown(
                #     """
                # # Fish Agent
                # 1. 此Demo为Fish Audio自研端到端语言模型Fish Agent 3B版本.
                # 2. 你可以在我们的官方仓库找到代码以及权重，但是相关内容全部基于 CC BY-NC-SA 4.0 许可证发布.
                # 3. Demo为早期灰度测试版本，推理速度尚待优化.
                # # 特色
                # 1. 该模型自动集成ASR与TTS部分，不需要外挂其它模型，即真正的端到端，而非三段式(ASR+LLM+TTS).
                # 2. 模型可以使用reference audio控制说话音色.
                # 3. 可以生成具有较强情感与韵律的音频.
                # """
                # )
                notes = gr.Markdown(
                    """
                    # Fish Agent
                    1. This demo is Fish Audio's self-researh end-to-end language model, Fish Agent version 3B.
                    2. You can find the code and weights in our official repo in [gitub](https://github.com/fishaudio/fish-speech) and [hugging face](https://huggingface.co/fishaudio/fish-agent-v0.1-3b), but the content is released under a CC BY-NC-SA 4.0 licence.
                    3. The demo is an early alpha test version, the inference speed needs to be optimised.
                    # Features
                    1. The model automatically integrates ASR and TTS parts, no need to plug-in other models, i.e., true end-to-end, not three-stage (ASR+LLM+TTS).
                    2. The model can use reference audio to control the speech timbre. 
                    3. The model can generate speech with strong emotion.
                """
                )

            # Right column (30%) for controls
            with gr.Column(scale=3):
                sys_audio_input = gr.Audio(
                    sources=["upload"],
                    type="numpy",
                    label="Give a timbre for your assistant",
                )
                sys_text_input = gr.Textbox(
                    label="What is your assistant's role?",
                    value="You are a voice assistant created by Fish Audio, offering end-to-end voice interaction for a seamless user experience. You are required to first transcribe the user's speech, then answer it in the following format: 'Question: [USER_SPEECH]\n\nAnswer: [YOUR_RESPONSE]\n'. You are required to use the following voice in this conversation.",
                    type="text",
                )
                audio_input = gr.Audio(
                    sources=["microphone"], type="numpy", label="Speak your message"
                )

                text_input = gr.Textbox(label="Or type your message", type="text")

                output_audio = gr.Audio(
                    label="Assistant's Voice",
                    streaming=True,
                    autoplay=True,
                    interactive=False,
                )

                send_button = gr.Button("Send", variant="primary")
                clear_button = gr.Button("Clear")

        # Event handlers
        audio_input.stop_recording(
            process_audio_input,
            inputs=[sys_audio_input, sys_text_input, audio_input, state, text_input],
            outputs=[chatbot, output_audio, audio_input, text_input],
            show_progress=True,
        )

        send_button.click(
            process_text_input,
            inputs=[sys_audio_input, sys_text_input, state, text_input],
            outputs=[chatbot, output_audio, audio_input, text_input],
            show_progress=True,
        )

        text_input.submit(
            process_text_input,
            inputs=[sys_audio_input, sys_text_input, state, text_input],
            outputs=[chatbot, output_audio, audio_input, text_input],
            show_progress=True,
        )

        clear_button.click(
            clear_fn,
            inputs=[],
            outputs=[chatbot, state, audio_input, output_audio, text_input],
        )

    return demo


if __name__ == "__main__":
    demo = create_demo()
    demo.launch(server_name="127.0.0.1", server_port=7860, share=True)
