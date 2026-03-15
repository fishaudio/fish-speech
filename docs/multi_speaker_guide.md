# Multi-Speaker TTS Generation Guide

Fish Speech supports multi-speaker text-to-speech generation through in-context learning using reference audio. This guide explains how to use native multi-speaker generation features.

## Overview

Multi-speaker generation uses reference audio to clone voices. The model learns the voice characteristics from the reference audio and synthesizes new speech in that voice.

**Key Concepts:**

- **Reference Audio**: A short audio clip (3-10 seconds) of the target voice
- **Reference Text**: The transcript of the reference audio
- **Reference ID**: A named collection of reference audios stored in the `references/` directory
- **In-Context Learning**: The model learns voice characteristics from references and applies them to new text

## Usage Methods

### Method 1: Direct Reference Audio (CLI)

Provide reference audio file(s) and their corresponding text transcripts directly:

```bash
python -m tools.api_client \
  --text "Hello, this is the text I want to synthesize." \
  --reference_audio /path/to/reference_audio.wav \
  --reference_text "This is the transcript of the reference audio." \
  --output output.wav
```

**Multiple References:**

You can provide multiple reference audio segments for better voice cloning:

```bash
python -m tools.api_client \
  --text "Hello, this is my synthesized speech." \
  --reference_audio ref1.wav ref2.wav ref3.wav \
  --reference_text "First reference text." "Second reference text" "Third reference text" \
  --output output.wav
```

**Parameters:**

- `--reference_audio`: Path(s) to reference audio file(s) (WAV, MP3, etc.)
- `--reference_text`: Corresponding transcript(s) for each reference audio
- `--output`: Output audio filename (default: "generated_audio")
- `--format`: Output format: wav, mp3, or flac (default: "wav")

### Method 2: Reference by ID (CLI)

Store reference audios in the `references/` directory and use them by ID:

1. **Create reference directory:**
```bash
mkdir -p references/my_speaker
```

2. **Add reference audio and transcript:**
```bash
cp /path/to/reference_audio.wav references/my_speaker/
echo "This is the transcript of the reference audio." > references/my_speaker/reference_audio.lab
```

3. **Use reference by ID:**
```bash
python -m tools.api_client \
  --text "Hello, this is synthesized speech in my voice." \
  --reference_id my_speaker \
  --output output.wav
```

**Advantages of Reference ID:**
- References are cached in memory for faster repeated use
- Easier to manage multiple voice profiles
- No need to specify paths each time

### Method 3: Python API (Direct)

Use the Fish Speech API directly in Python:

```python
from fish_speech.utils.schema import ServeReferenceAudio, ServeTTSRequest
from fish_speech.utils.file import audio_to_bytes, read_ref_text

# Load reference audio and text
reference_audio = audio_to_bytes("/path/to/reference.wav")
reference_text = read_ref_text("/path/to/reference.lab")

# Create request
request = ServeTTSRequest(
    text="Hello, this is synthesized speech.",
    references=[
        ServeReferenceAudio(
            audio=reference_audio,
            text=reference_text
        )
    ],
    chunk_length=200,
    format="wav"
)

# Send to inference engine (example with server)
import requests
import ormsgpack

response = requests.post(
    "http://localhost:8080/v1/tts",
    params={"format": "msgpack"},
    data=ormsgpack.packb(request, option=ormsgpack.OPT_SERIALIZE_PYDANTIC)
)
```

### Method 4: REST API (HTTP)

Send a POST request to the `/v1/tts` endpoint:

```python
import base64
import requests

# Prepare reference audio
with open("/path/to/reference.wav", "rb") as f:
    audio_bytes = f.read()
    audio_base64 = base64.b64encode(audio_bytes).decode()

# Create request payload
payload = {
    "text": "Hello, this is synthesized speech.",
    "references": [
        {
            "audio": audio_base64,
            "text": "This is the transcript of the reference audio."
        }
    ],
    "chunk_length": 200,
    "format": "wav",
    "max_new_tokens": 1024
}

# Send request
response = requests.post(
    "http://localhost:8080/v1/tts",
    json=payload
)

# Save audio
with open("output.wav", "wb") as f:
    f.write(response.content)
```

## Important Notes

### Reference Audio Quality

- **Duration**: 3-10 seconds is ideal
- **Quality**: Clean audio without background noise
- **Content**: Should contain natural speech with clear pronunciation
- **Format**: WAV, MP3, or other common formats (automatically converted)

### Reference Text Requirements

- **Accuracy**: Must be the exact transcript of the reference audio
- **Encoding**: Text files should be UTF-8 encoded
- **Format**: Use `.lab` extension for reference text files

### Memory Caching

For repeated use of the same references, enable memory caching:

```bash
python -m tools.api_client \
  --text "Your text here" \
  --reference_id my_speaker \
  --use_memory_cache on \
  --output output.wav
```

This caches the encoded reference audio in memory for faster subsequent requests.

## Native Multi-Speaker Features

### Voice Cloning Accuracy

The system creates a voice embedding from reference audio. More reference segments generally improve accuracy:

- **Single reference**: Basic voice cloning
- **3-5 references**: Better voice consistency
- **10+ references**: Optimal for style and prosody capture

### Prosody Control with Natural Language

While native emotion tags like `<happy>` or `<sad>` are not supported (see Issue #1180), you can control prosody using natural language descriptions:

```python
# Instead of: "<happy>Hello world"
# Use: "Hello world" with reference audio that sounds happy

# Or add descriptive text:
request = ServeTTSRequest(
    text="She whispered softly: hello there",
    # ... references ...
)
```

### Seed for Reproducibility

Use the `seed` parameter to generate deterministic results:

```bash
python -m tools.api_client \
  --text "Hello world" \
  --reference_id my_speaker \
  --seed 42 \
  --output output.wav
```

**Note**: Seed ensures reproducible synthesis but cannot fix timbre. Voice characteristics depend entirely on the reference audio.

## Troubleshooting

### Voice Doesn't Sound Like Reference

**Possible causes:**
- Reference audio is too short or poor quality
- Reference text doesn't match the audio content
- Background noise in reference audio
- Using too few reference samples

**Solutions:**
- Use longer, cleaner reference audio (5-10 seconds)
- Verify reference text accuracy
- Record in quiet environment
- Provide multiple reference samples

### Audio Artifacts

**Possible causes:**
- Chunk length too small or too large
- Punctuation issues in text
- Reference audio has artifacts

**Solutions:**
- Adjust `--chunk_length` (150-250 is usually good)
- Add proper punctuation to text
- Clean up reference audio

### Slow Inference

**Possible causes:**
- Large reference audio files
- Too many reference segments
- Memory cache disabled for repeated uses

**Solutions:**
- Use shorter reference segments
- Limit to 3-5 best reference samples
- Enable `--use_memory_cache on`

## Advanced Usage

### Multi-Language Code-Switching

If your reference audio contains code-switching (mixing languages), the model can preserve this ability:

```bash
# Reference contains English and Chinese
python -m tools.api_client \
  --text "Hello world, 你好世界" \
  --reference_audio bilingual_ref.wav \
  --reference_text "Hello 大家好" \
  --output output.wav
```

### Style Transfer

To capture specific speaking styles (e.g., storytelling, news reading, conversational):

1. Use reference audio with the desired style
2. Provide multiple examples of the same style
3. The model learns the speaking patterns and applies them

### Fine-Grained Control

For precise control, you can:

1. **Use reference_id for predefined voices**
2. **Mix multiple reference styles**
3. **Adjust generation parameters**:
   - `--temperature`: Control randomness (0.1-1.0)
   - `--top_p`: Control diversity (0.1-1.0)
   - `--repetition_penalty`: Prevent repetition (1.0-2.0)

## API Reference

### ServeTTSRequest Schema

```python
class ServeTTSRequest(BaseModel):
    text: str                                          # Text to synthesize
    chunk_length: int = 200                           # Segment length in characters
    format: Literal["wav", "pcm", "mp3"] = "wav"     # Output format
    references: list[ServeReferenceAudio] = []       # Reference audio samples
    reference_id: str | None = None                  # Named reference collection
    seed: int | None = None                          # Random seed
    use_memory_cache: Literal["on", "off"] = "off"   # Cache references in memory
    # ... generation parameters ...
```

### ServeReferenceAudio Schema

```python
class ServeReferenceAudio(BaseModel):
    audio: bytes    # Reference audio data (base64 encoded in JSON)
    text: str       # Corresponding transcript
```

## Related Issues

- Issue #1188: Segment model loading documentation
- Issue #1183: --compile option audio distortion
- Issue #1180: Emotion tags documentation
