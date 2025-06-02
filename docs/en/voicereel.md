# VoiceReel API Usage

This page describes how to use the VoiceReel multi-speaker TTS API.

## Register a Speaker

Send a POST request to `/v1/speakers` with a reference audio file and matching transcript:

```bash
curl -X POST $SERVER/v1/speakers \
  -F name=alice \
  -F lang=en \
  -F reference_audio=@alice.wav \
  -F reference_script=@alice.txt
```

The API returns a job ID and temporary speaker ID:

```json
{ "job_id": "job_123", "speaker_id": "spk_tmp" }
```

## Synthesize a Script

Provide a JSON script containing speaker IDs and text segments:

```bash
curl -X POST $SERVER/v1/synthesize \
  -H 'Content-Type: application/json' \
  -d '{
    "script": [
      {"speaker_id": "spk_tmp", "text": "Hello."},
      {"speaker_id": "spk2", "text": "Welcome."}
    ],
    "caption_format": "vtt"
  }'
```

The response includes a job ID. Poll `/v1/jobs/{id}` to retrieve the audio and caption URLs.

## Check Job Status

```bash
curl $SERVER/v1/jobs/job_123
```

If the job succeeded, the response will contain download links for the audio and captions.

## Error Codes

The API returns standardized error responses. The table below lists the codes defined in the PRD:

| Code | Meaning | Description |
| --- | --- | --- |
| 400 `INVALID_INPUT` | invalid parameters | missing fields or malformed JSON |
| 401 `UNAUTHORIZED` | authentication failure | API key mismatch or bad HMAC |
| 404 `NOT_FOUND` | resource missing | wrong `speaker_id` or `job_id` |
| 413 `PAYLOAD_TOO_LARGE` | upload too big | reference audio > 30&nbsp;MB |
| 422 `INSUFFICIENT_REF` | reference too short | at least 30&nbsp;sec required |
| 500 `INTERNAL_ERROR` | server error | unexpected exception |
