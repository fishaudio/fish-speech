# FunASR Command Line Interface

This tool provides a command-line interface for separating vocals from instrumental tracks, converting videos to audio, and performing speech-to-text transcription on the resulting audio files.

## Requirements

- Python >= 3.10
- PyTorch <= 2.3.1
- ffmpeg, pydub, audio-separator[gpu].

## Installation

Install the required packages:

```bash
pip install -e .[stable]
```

Make sure you have `ffmpeg` installed and available in your `PATH`.

## Usage

### Basic Usage

To run the tool with default settings:

```bash
python tools/sensevoice/fun_asr.py --audio-dir <audio_directory> --save-dir <output_directory>
```

## Options

|          Option           |                                  Description                                  |
| :-----------------------: | :---------------------------------------------------------------------------: |
|        --audio-dir        |                  Directory containing audio or video files.                   |
|        --save-dir         |                   Directory to save processed audio files.                    |
|         --device          |         Device to use for processing. Options: cuda (default) or cpu.         |
|        --language         |                Language of the transcription. Default is auto.                |
| --max_single_segment_time | Maximum duration of a single audio segment in milliseconds. Default is 20000. |
|          --punc           |                        Enable punctuation prediction.                         |
|         --denoise         |                  Enable noise reduction (vocal separation).                   |

## Example

To process audio files in the directory `path/to/audio` and save the output to `path/to/output`, with punctuation and noise reduction enabled:

```bash
python tools/sensevoice/fun_asr.py --audio-dir path/to/audio --save-dir path/to/output --punc --denoise
```

## Additional Notes

- The tool supports `both audio and video files`. Videos will be converted to audio automatically.
- If the `--denoise` option is used, the tool will perform vocal separation to isolate the vocals from the instrumental tracks.
- The script will automatically create necessary directories in the `--save-dir`.

## Troubleshooting

If you encounter any issues, make sure all dependencies are correctly installed and configured. For more detailed troubleshooting, refer to the documentation of each dependency.
