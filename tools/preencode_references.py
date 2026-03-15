#!/usr/bin/env python3
"""
Pre-encode reference audio to .codes.pt so the server skips DAC encode at runtime.
Reads a folder of *.wav + *.txt (same stem), runs DAC encoder once per file (sequential).

Without --ref-id: one reference per file, ID = stem.  ru.wav + ru.txt -> references/ru/ru.codes.pt, ru.lab
With --ref-id: all files under one reference.  -> references/<ref_id>/<stem>.codes.pt, <stem>.lab

With --upload: after writing each reference, POST to server /v1/references/add_encoded. Server skips
write if content hash matches (unchanged). Requires server to be running.

  python tools/preencode_references.py --input-dir ./data/voice_references
  python tools/preencode_references.py --input-dir ./data/voice_references --upload --server-url http://127.0.0.1:8080
"""

import urllib.error
import urllib.request
from pathlib import Path

import click
import torch
import torchaudio
from loguru import logger

from fish_speech.models.dac.inference import load_model as load_dac_model
from fish_speech.utils.file import AUDIO_EXTENSIONS


def _upload_encoded(
    server_url: str,
    ref_id: str,
    codes_path: Path,
    lab_path: Path,
    stem: str | None = None,
) -> None:
    """POST pre-encoded reference to server. Server skips if hash matches."""
    stem = stem or ref_id
    codes_bytes = codes_path.read_bytes()
    lab_text = lab_path.read_text(encoding="utf-8")
    url = f"{server_url.rstrip('/')}/v1/references/add_encoded"
    boundary = "----form-boundary----"
    body_parts = [
        f'--{boundary}\r\nContent-Disposition: form-data; name="id"\r\n\r\n{ref_id}\r\n',
        f'--{boundary}\r\nContent-Disposition: form-data; name="stem"\r\n\r\n{stem}\r\n',
        f'--{boundary}\r\nContent-Disposition: form-data; name="codes"; filename="{stem}.codes.pt"\r\nContent-Type: application/octet-stream\r\n\r\n',
        codes_bytes,
        b"\r\n",
        f'--{boundary}\r\nContent-Disposition: form-data; name="lab"; filename="{stem}.lab"\r\nContent-Type: text/plain\r\n\r\n',
        lab_text.encode("utf-8"),
        b"\r\n",
        f"--{boundary}--\r\n",
    ]
    body = b""
    for p in body_parts:
        body = body + (p if isinstance(p, bytes) else p.encode("utf-8"))
    req = urllib.request.Request(url, data=body, method="POST")
    req.add_header("Content-Type", f"multipart/form-data; boundary={boundary}")
    req.add_header("Content-Length", str(len(body)))
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            data = r.read().decode("utf-8")
            logger.info("Upload {}/{}: {}", ref_id, stem, data)
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        logger.warning("Upload {}/{} failed: {} {}", ref_id, stem, e.code, body)
    except Exception as e:
        logger.warning("Upload {}/{} error: {}", ref_id, stem, e)


@click.command()
@click.option(
    "--input-dir",
    "-i",
    type=click.Path(exists=True, path_type=Path),
    required=False,
    default=None,
    help="Folder with pairs: <stem>.wav + <stem>.txt (or .lab). Not needed with --upload-only.",
)
@click.option(
    "--ref-id",
    "-r",
    default=None,
    help="Reference ID. If omitted, ID = stem per file (ru.wav -> references/ru/). If set, all files go under <output-dir>/<ref_id>/",
)
@click.option(
    "--output-dir",
    "-o",
    default="references",
    type=click.Path(path_type=Path),
    help="Base dir for references (default: references). Result: <output-dir>/<ref_id>/",
)
@click.option("--config-name", default="modded_dac_vq", help="DAC config name")
@click.option(
    "--checkpoint-path",
    default="checkpoints/s2-pro/codec.pth",
    type=click.Path(path_type=Path),
    help="DAC checkpoint",
)
@click.option(
    "--device",
    "-d",
    default="cpu",
    type=click.Choice(["cpu", "cuda", "cuda:0"]),
    help="Device for encode (cpu = no VRAM spike, one file at a time)",
)
@click.option(
    "--upload",
    is_flag=True,
    help="After encoding, upload each reference to server (POST /v1/references/add_encoded). Server skips if hash matches.",
)
@click.option(
    "--server-url",
    default="http://127.0.0.1:8080",
    help="Server base URL for --upload (default: http://127.0.0.1:8080)",
)
@click.option(
    "--upload-only",
    is_flag=True,
    help="Only upload existing references/ (no encoding). Scan output-dir for *.codes.pt + *.lab and POST each. No DAC, no input-dir.",
)
def main(
    input_dir: Path,
    ref_id: str | None,
    output_dir: Path,
    config_name: str,
    checkpoint_path: Path,
    device: str,
    upload: bool,
    server_url: str,
    upload_only: bool,
) -> None:
    base_out = Path(output_dir).resolve()

    if upload_only:
        # Scan base_out for ref_id dirs, each with *.codes.pt + *.lab; upload without encoding
        if not base_out.exists():
            logger.error("Output dir %s not found", base_out)
            raise SystemExit(1)
        for ref_dir in sorted(base_out.iterdir()):
            if not ref_dir.is_dir():
                continue
            ref_id_name = ref_dir.name
            for codes_path in sorted(ref_dir.glob("*.codes.pt")):
                stem = (
                    codes_path.stem.removesuffix(".codes")
                    if codes_path.stem.endswith(".codes")
                    else codes_path.stem
                )
                lab_path = ref_dir / f"{stem}.lab"
                if not lab_path.exists():
                    logger.warning("No %s for %s, skip", lab_path.name, codes_path.name)
                    continue
                logger.info("Upload only: %s/%s", ref_id_name, stem)
                _upload_encoded(
                    server_url, ref_id_name, codes_path, lab_path, stem=stem
                )
        logger.info("Upload-only done for %s", base_out)
        return

    if input_dir is None:
        raise click.UsageError("--input-dir required unless --upload-only")
    input_dir = input_dir.resolve()

    # Find stems that have both .wav and .txt (or .lab)
    wavs = {
        p.stem: p
        for p in input_dir.iterdir()
        if p.suffix.lower() in AUDIO_EXTENSIONS and p.is_file()
    }
    stems = []
    for stem, wav_path in sorted(wavs.items()):
        txt = input_dir / f"{stem}.txt"
        lab = input_dir / f"{stem}.lab"
        if txt.exists():
            text_path = txt
        elif lab.exists():
            text_path = lab
        else:
            logger.warning("No .txt/.lab for {}, skipping", stem)
            continue
        stems.append((stem, wav_path, text_path))

    if not stems:
        logger.error("No (wav + txt/lab) pairs in {}", input_dir)
        raise SystemExit(1)

    # One ref_id per file (ID = stem) or one folder for all
    use_stem_as_id = ref_id is None
    if use_stem_as_id:
        logger.info("Output: {} (one reference per file, ID = stem)", base_out)
    else:
        ref_folder = base_out / ref_id
        ref_folder.mkdir(parents=True, exist_ok=True)
        logger.info("Output: {}", ref_folder)

    logger.info("Loading DAC once (device={})...", device)
    model = load_dac_model(
        config_name=config_name,
        checkpoint_path=str(checkpoint_path),
        device=device,
    )

    for stem, wav_path, text_path in stems:
        ref_folder = base_out / (stem if use_stem_as_id else ref_id)
        ref_folder.mkdir(parents=True, exist_ok=True)
        logger.info(
            "Encoding {} -> {} (id={})",
            stem,
            ref_folder,
            stem if use_stem_as_id else ref_id,
        )
        audio, sr = torchaudio.load(str(wav_path))
        if audio.shape[0] > 1:
            audio = audio.mean(0, keepdim=True)
        audio = torchaudio.functional.resample(audio, sr, model.sample_rate)
        audios = audio[None].to(device)  # (1, 1, T)
        audio_lengths = torch.tensor([audios.shape[2]], device=device, dtype=torch.long)
        with torch.no_grad():
            indices, _ = model.encode(audios, audio_lengths)
        if indices.ndim == 3:
            indices = indices[0]  # (num_codebooks, T)
        codes_path = ref_folder / f"{stem}.codes.pt"
        torch.save(indices.cpu(), codes_path)
        lab_text = text_path.read_text(encoding="utf-8")
        lab_path = ref_folder / f"{stem}.lab"
        lab_path.write_text(lab_text, encoding="utf-8")

        if upload:
            ref_id_for_upload = stem if use_stem_as_id else ref_id
            stem_for_upload = stem
            _upload_encoded(
                server_url,
                ref_id_for_upload,
                codes_path,
                lab_path,
                stem=stem_for_upload,
            )

    logger.info("Done. {} reference(s) in {}", len(stems), base_out)


if __name__ == "__main__":
    main()
