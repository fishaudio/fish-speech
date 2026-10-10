"""Run the S1 Mini version pinned by the Pinokio OpenAudio launcher."""

import argparse
import atexit
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
RUNTIME = ROOT / ".runtime" / "openaudio"
CHECKPOINT = ROOT / "checkpoints" / "openaudio-s1-mini"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["auto", "mps", "cpu"], default="auto")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    if not RUNTIME.joinpath(".project-root").exists():
        raise SystemExit("The Pinokio runtime is missing. See LOCAL_SETUP.md.")
    for name, size in {"model.pth": 1735122974, "codec.pth": 1871099728}.items():
        path = CHECKPOINT / name
        if not path.exists() or path.stat().st_size != size:
            raise SystemExit("Model download is incomplete. Run 'Download Model.command'.")

    pid_file = ROOT / ".local-voice" / "server.pid"
    pid_file.parent.mkdir(exist_ok=True)
    pid_file.write_text(str(os.getpid()) + "\n")

    def remove_pid():
        if pid_file.exists() and pid_file.read_text().strip() == str(os.getpid()):
            pid_file.unlink()

    atexit.register(remove_pid)

    os.chdir(RUNTIME)
    sys.path.insert(0, str(RUNTIME))
    os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "huggingface"))
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))
    os.environ.setdefault("GRADIO_TEMP_DIR", str(ROOT / ".cache" / "gradio"))
    os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

    import numpy as np
    import soundfile as sf
    import torch
    from loguru import logger
    from fish_speech.inference_engine import TTSInferenceEngine
    from fish_speech.models.dac.inference import load_model as load_decoder
    from fish_speech.models.text2semantic.inference import launch_thread_safe_queue
    from fish_speech.utils.schema import ServeTTSRequest
    import tools.webui as webui
    from tools.webui.inference import get_inference_wrapper

    device = args.device
    if device == "auto":
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    if device == "mps" and not torch.backends.mps.is_available():
        raise SystemExit("Apple GPU is unavailable to this process. Run the launcher in Terminal.")
    logger.info("Loading S1 Mini on {} from {}", device, CHECKPOINT)
    precision = torch.bfloat16
    llama_queue = launch_thread_safe_queue(
        checkpoint_path=CHECKPOINT, device=device, precision=precision, compile=False
    )
    decoder = load_decoder("modded_dac_vq", CHECKPOINT / "codec.pth", device=device)
    engine = TTSInferenceEngine(
        llama_queue=llama_queue, decoder_model=decoder, compile=False, precision=precision
    )

    logger.info("Generating a short startup sample...")
    audio = None
    for result in engine.inference(ServeTTSRequest(
        text="Hello! Your local voice model is ready.", references=[],
        max_new_tokens=256, chunk_length=200, seed=42,
        top_p=0.8, repetition_penalty=1.1, temperature=0.8,
    )):
        if result.code == "error":
            raise RuntimeError(f"Startup speech generation failed: {result.error}")
        if result.code == "final":
            audio = result.audio
    if audio is None:
        raise RuntimeError("Startup speech generation returned no audio.")
    sample_rate, samples = audio
    if samples.size == 0 or not np.isfinite(samples).all() or np.max(np.abs(samples)) < 1e-6:
        raise RuntimeError("Startup speech generation returned invalid or silent audio.")
    output = ROOT / "results" / "setup-test.wav"
    output.parent.mkdir(exist_ok=True)
    sf.write(output, samples, sample_rate)
    logger.info("Speech verified: {:.2f} seconds at {} Hz; saved to {}",
                len(samples) / sample_rate, sample_rate, output)
    if args.smoke_test:
        return

    webui.HEADER_MD = webui.HEADER_MD.replace(
        "# Fish Speech", "# Fish Speech · S1 Mini"
    ).replace(
        "https://huggingface.co/fishaudio/fish-speech-1.5",
        "https://huggingface.co/fishaudio/s1-mini",
    )
    app = webui.build_app(get_inference_wrapper(engine), "light")
    app.title = "Fish Speech · S1 Mini"
    app.launch(server_name="127.0.0.1", server_port=args.port,
               inbrowser=not args.no_browser, share=False)


if __name__ == "__main__":
    main()
