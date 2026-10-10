# Local Fish Speech / OpenAudio S1 Mini

Double-click **Start Fish Speech.command** in Finder. It uses the Apple GPU,
generates a startup sample, and opens **http://127.0.0.1:7860**. Keep its Terminal
window open while using the model; press **Control-C** to stop it.
You can also double-click **Stop Fish Speech.command** to stop the server,
including the server started during setup.

Enter text and click **Generate**. To specify a voice, open **Reference Audio**,
upload a clear 5–10 second recording, and enter its exact transcript. Leave
Reference ID empty when uploading audio. Generated audio can be downloaded from
the audio player.

The startup sample is saved as `results/setup-test.wav`.
The server started during setup writes its output to `logs/webui.log`.

## Installed configuration

- Pinokio launcher: https://github.com/pinokiofactory/openaudio
- Matching Fish Speech source revision: `87fffd0`, in `.runtime/openaudio`.
- Model repository used by that launcher: `cocktailpeanut/oa`, revision
  `be35f06bb5c7604fc2197f631bc3cd4e3286af30`.
- Weights: `checkpoints/openaudio-s1-mini` (about 3.6 GB).
- Python: existing project `venv`, Python 3.10.
- PyTorch and torchaudio: 2.8.0; NumPy: 2.2.6; Gradio: 6.17.3.
- Protobuf: 5.29.6. This is an intentional override of descript-audiotools'
  outdated `<3.20` dependency, required by Fish Speech's generated protobuf code.
- Apple Metal (`mps`), bfloat16, compilation disabled.
- The web interface listens only on `127.0.0.1`; public sharing is disabled.
- Inference uses local model files without a Hugging Face login or cloud API.

Your original checkout is retained. The launcher loads the matching S1 Mini
source from `.runtime/openaudio` rather than the newer S2 source in the root.

## Terminal commands

Run from this project directory:

```sh
./"Start Fish Speech.command"
./venv/bin/python local_voice.py --smoke-test
./venv/bin/python local_voice.py --device cpu --port 7861
```

If a download is interrupted, double-click **Download Model.command** to resume.
Model access and use are subject to the model's license; see its model card:
https://huggingface.co/fishaudio/s1-mini
