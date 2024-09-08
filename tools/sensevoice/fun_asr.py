import gc
import os
import re

from audio_separator.separator import Separator

os.environ["MODELSCOPE_CACHE"] = "./.cache/funasr"
os.environ["UVR5_CACHE"] = "./.cache/uvr5-models"
import json
import subprocess
from pathlib import Path

import click
import torch
from loguru import logger
from pydub import AudioSegment
from silero_vad import get_speech_timestamps, load_silero_vad, read_audio
from tqdm import tqdm

from tools.file import AUDIO_EXTENSIONS, VIDEO_EXTENSIONS, list_files
from tools.sensevoice.auto_model import AutoModel


def uvr5_cli(
    audio_dir: Path,
    output_folder: Path,
    audio_files: list[Path] | None = None,
    output_format: str = "flac",
    model: str = "BS-Roformer-Viperx-1297.ckpt",
):
    # ["BS-Roformer-Viperx-1297.ckpt", "BS-Roformer-Viperx-1296.ckpt", "BS-Roformer-Viperx-1053.ckpt", "Mel-Roformer-Viperx-1143.ckpt"]
    sepr = Separator(
        model_file_dir=os.environ["UVR5_CACHE"],
        output_dir=output_folder,
        output_format=output_format,
    )
    dictmodel = {
        "BS-Roformer-Viperx-1297.ckpt": "model_bs_roformer_ep_317_sdr_12.9755.ckpt",
        "BS-Roformer-Viperx-1296.ckpt": "model_bs_roformer_ep_368_sdr_12.9628.ckpt",
        "BS-Roformer-Viperx-1053.ckpt": "model_bs_roformer_ep_937_sdr_10.5309.ckpt",
        "Mel-Roformer-Viperx-1143.ckpt": "model_mel_band_roformer_ep_3005_sdr_11.4360.ckpt",
    }
    roformer_model = dictmodel[model]
    sepr.load_model(roformer_model)
    if audio_files is None:
        audio_files = list_files(
            path=audio_dir, extensions=AUDIO_EXTENSIONS, recursive=True
        )
    total_files = len(audio_files)

    print(f"{total_files} audio files found")

    res = []
    for audio in tqdm(audio_files, desc="Denoising: "):
        file_path = str(audio_dir / audio)
        sep_out = sepr.separate(file_path)
        if isinstance(sep_out, str):
            res.append(sep_out)
        elif isinstance(sep_out, list):
            res.extend(sep_out)
    del sepr
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return res, roformer_model


def get_sample_rate(media_path: Path):
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "quiet",
            "-print_format",
            "json",
            "-show_streams",
            str(media_path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    media_info = json.loads(result.stdout)
    for stream in media_info.get("streams", []):
        if stream.get("codec_type") == "audio":
            return stream.get("sample_rate")
    return "44100"  # Default sample rate if not found


def convert_to_mono(src_path: Path, out_path: Path, out_fmt: str = "wav"):
    sr = get_sample_rate(src_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if src_path.resolve() == out_path.resolve():
        output = str(out_path.with_stem(out_path.stem + f"_{sr}"))
    else:
        output = str(out_path)
    subprocess.run(
        [
            "ffmpeg",
            "-loglevel",
            "error",
            "-i",
            str(src_path),
            "-acodec",
            "pcm_s16le" if out_fmt == "wav" else "flac",
            "-ar",
            sr,
            "-ac",
            "1",
            "-y",
            output,
        ],
        check=True,
    )
    return out_path


def convert_video_to_audio(video_path: Path, audio_dir: Path):
    cur_dir = audio_dir / video_path.relative_to(audio_dir).parent
    vocals = [
        p
        for p in cur_dir.glob(f"{video_path.stem}_(Vocals)*.*")
        if p.suffix in AUDIO_EXTENSIONS
    ]
    if len(vocals) > 0:
        return vocals[0]
    audio_path = cur_dir / f"{video_path.stem}.wav"
    convert_to_mono(video_path, audio_path)
    return audio_path


@click.command()
@click.option("--audio-dir", required=True, help="Directory containing audio files")
@click.option(
    "--save-dir", required=True, help="Directory to save processed audio files"
)
@click.option("--device", default="cuda", help="Device to use [cuda / cpu]")
@click.option("--language", default="auto", help="Language of the transcription")
@click.option(
    "--max_single_segment_time",
    default=20000,
    type=int,
    help="Maximum of Output single audio duration(ms)",
)
@click.option("--fsmn-vad/--silero-vad", default=False)
@click.option("--punc/--no-punc", default=False)
@click.option("--denoise/--no-denoise", default=False)
@click.option("--save_emo/--no_save_emo", default=False)
def main(
    audio_dir: str,
    save_dir: str,
    device: str,
    language: str,
    max_single_segment_time: int,
    fsmn_vad: bool,
    punc: bool,
    denoise: bool,
    save_emo: bool,
):

    audios_path = Path(audio_dir)
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    video_files = list_files(
        path=audio_dir, extensions=VIDEO_EXTENSIONS, recursive=True
    )
    v2a_files = [convert_video_to_audio(p, audio_dir) for p in video_files]

    if denoise:
        VOCAL = "_(Vocals)"
        original_files = [
            p
            for p in audios_path.glob("**/*")
            if p.suffix in AUDIO_EXTENSIONS and VOCAL not in p.stem
        ]

        _, cur_model = uvr5_cli(
            audio_dir=audio_dir, output_folder=audio_dir, audio_files=original_files
        )
        need_remove = [p for p in audios_path.glob("**/*(Instrumental)*")]
        need_remove.extend(original_files)
        for _ in need_remove:
            _.unlink()
        vocal_files = [
            p
            for p in audios_path.glob("**/*")
            if p.suffix in AUDIO_EXTENSIONS and VOCAL in p.stem
        ]
        for f in vocal_files:
            fn, ext = f.stem, f.suffix

            v_pos = fn.find(VOCAL + "_" + cur_model.split(".")[0])
            if v_pos != -1:
                new_fn = fn[: v_pos + len(VOCAL)]
                new_f = f.with_name(new_fn + ext)
                f = f.rename(new_f)
                convert_to_mono(f, f, "flac")
                f.unlink()

    audio_files = list_files(
        path=audio_dir, extensions=AUDIO_EXTENSIONS, recursive=True
    )

    logger.info("Loading / Downloading Funasr model...")

    model_dir = "iic/SenseVoiceSmall"

    vad_model = "fsmn-vad" if fsmn_vad else None
    vad_kwargs = {"max_single_segment_time": max_single_segment_time}
    punc_model = "ct-punc" if punc else None

    manager = AutoModel(
        model=model_dir,
        trust_remote_code=False,
        vad_model=vad_model,
        vad_kwargs=vad_kwargs,
        punc_model=punc_model,
        device=device,
    )

    if not fsmn_vad and vad_model is None:
        vad_model = load_silero_vad()

    logger.info("Model loaded.")

    pattern = re.compile(r"_\d{3}\.")

    for file_path in tqdm(audio_files, desc="Processing audio file"):

        if pattern.search(file_path.name):
            # logger.info(f"Skipping {file_path} as it has already been processed.")
            continue

        file_stem = file_path.stem
        file_suffix = file_path.suffix

        rel_path = Path(file_path).relative_to(audio_dir)
        (save_path / rel_path.parent).mkdir(parents=True, exist_ok=True)

        audio = AudioSegment.from_file(file_path)

        cfg = dict(
            cache={},
            language=language,  # "zh", "en", "yue", "ja", "ko", "nospeech"
            use_itn=False,
            batch_size_s=60,
        )

        if fsmn_vad:
            elapsed, vad_res = manager.vad(input=str(file_path), **cfg)
        else:
            wav = read_audio(
                str(file_path)
            )  # backend (sox, soundfile, or ffmpeg) required!
            audio_key = file_path.stem
            audio_val = []
            speech_timestamps = get_speech_timestamps(
                wav,
                vad_model,
                max_speech_duration_s=max_single_segment_time // 1000,
                return_seconds=True,
            )

            audio_val = [
                [int(timestamp["start"] * 1000), int(timestamp["end"] * 1000)]
                for timestamp in speech_timestamps
            ]
            vad_res = []
            vad_res.append(dict(key=audio_key, value=audio_val))

        res = manager.inference_with_vadres(
            input=str(file_path), vad_res=vad_res, **cfg
        )

        for i, info in enumerate(res):
            [start_ms, end_ms] = info["interval"]
            text = info["text"]
            emo = info["emo"]
            sliced_audio = audio[start_ms:end_ms]
            audio_save_path = (
                save_path / rel_path.parent / f"{file_stem}_{i:03d}{file_suffix}"
            )
            sliced_audio.export(audio_save_path, format=file_suffix[1:])
            print(f"Exported {audio_save_path}: {text}")

            transcript_save_path = (
                save_path / rel_path.parent / f"{file_stem}_{i:03d}.lab"
            )
            with open(
                transcript_save_path,
                "w",
                encoding="utf-8",
            ) as f:
                f.write(text)

            if save_emo:
                emo_save_path = save_path / rel_path.parent / f"{file_stem}_{i:03d}.emo"
                with open(
                    emo_save_path,
                    "w",
                    encoding="utf-8",
                ) as f:
                    f.write(emo)

        if audios_path.resolve() == save_path.resolve():
            file_path.unlink()


if __name__ == "__main__":
    main()
    exit(0)
    from funasr.utils.postprocess_utils import rich_transcription_postprocess

    # Load the audio file
    audio_path = Path(r"D:\PythonProject\ok\1_output_(Vocals).wav")
    model_dir = "iic/SenseVoiceSmall"
    m, kwargs = SenseVoiceSmall.from_pretrained(model=model_dir, device="cuda:0")
    m.eval()

    res = m.inference(
        data_in=f"{kwargs['model_path']}/example/zh.mp3",
        language="auto",  # "zh", "en", "yue", "ja", "ko", "nospeech"
        use_itn=False,
        ban_emo_unk=False,
        **kwargs,
    )

    print(res)
    text = rich_transcription_postprocess(res[0][0]["text"])
    print(text)
