# Inference API Documentation

## Model Loading and Segment Processing

### Segment-Based Processing (Chunked Inference)

Fish Speech supports segment-based processing for memory-efficient handling of long texts. This feature automatically breaks down long input texts into smaller chunks/segments.

#### How It Works

The inference engine processes text in segments when `chunk_length > 0`. Each segment contains approximately `chunk_length` characters and is processed independently before being concatenated into the final audio output.

**Key parameters in `ServeTTSRequest`:**

- `chunk_length` (int, default: 200): Controls segment size in characters
  - Range: 100-300 characters
  - Set to 0 to disable segment processing
  - When > 0, enables segment-based processing

**Code flow:**

1. **Request Initiation**: The request is sent to `TTSInferenceEngine.inference()`
2. **Segment Detection**: Based on `chunk_length > 0` and `iterative_prompt` flag
3. **Processing Loop**:
   - Each segment is processed through the LLAMA model
   - Audio is generated via `get_audio_segment()`
   - Results are yielded with `code="segment"`
4. **Final Assembly**: All segments are concatenated into final audio

**Key files:**

- `fish_speech/utils/schema.py`: Defines `ServeTTSRequest` with `chunk_length` parameter
- `fish_speech/inference_engine/__init__.py`: Implements segment processing in `inference()` method
- `tools/server/inference.py`: Handles segment results in API wrapper

**Example usage:**

```python
from fish_speech.utils.schema import ServeTTSRequest

# Enable segment processing with 200-char chunks (default)
request = ServeTTSRequest(
    text="Your long text here...",
    chunk_length=200
)

# Or disable segment processing
request = ServeTTSRequest(
    text="Your text here...",
    chunk_length=0
)
```

### Model Loading Architecture

The inference engine loads and manages two main models:

1. **LLAMA Model**: Loaded via `llama_queue` for text-to-semantic token generation
2. **Decoder Model (DAC)**: Loaded via `decoder_model` for VQ token-to-audio decoding

**Initialization:**

```python
engine = TTSInferenceEngine(
    llama_queue=llama_queue,
    decoder_model=decoder_model,
    precision=torch.float16,
    compile=False
)
```

**Model configuration:**

- `precision`: Controls computation precision (e.g., `torch.float16`, `torch.bfloat16`)
- `compile`: Enables/disables torch.compile optimization

**Audio segment generation:**

The `get_audio_segment()` method in `TTSInferenceEngine`:
1. Takes VQ tokens from LLAMA model output
2. Decodes them to audio via `decode_vq_tokens()`
3. Returns numpy audio array for each segment

### REST API Usage

**Endpoint**: `POST /v1/tts`

**Request body:**

```json
{
  "text": "Text to synthesize",
  "chunk_length": 200,
  "format": "wav",
  "references": []
}
```

**Segment processing behavior:**

- When `chunk_length` > 0: Text is processed in segments
- When `chunk_length` = 0: Text is processed in single pass
- Each segment is yielded as it's generated in streaming mode

### Common Issues

**Audio artifacts at segment boundaries:**
- Ensure proper chunk_length (100-300 chars recommended)
- Check reference audio quality if using voice cloning
- Verify decoder model is loaded correctly

**Memory issues with long texts:**
- Reduce chunk_length for more frequent memory cleanup
- Enable streaming mode for incremental processing
- Check GPU memory availability

### CLI Usage

```bash
# With segment processing (default)
python -m tools.api_client --text "long text here" --chunk_length 200

# Without segment processing
python -m tools.api_client --text "text here" --chunk_length 0
```
