Docker image for running fish-speech v.1.5 with an API server. The container takes about a minute to start up with the --compile flag on the API server on an RTX4090. The model uses about 2.7GB VRAM.

# Build and Run the Container

`docker build . -t airia-tts-fishspeech`

`docker run --rm -p 8080:8080 --name airia-tts-fishspeech --gpus=all airia-tts-fishspeech`


# API Usage

Sample API call:
```
  curl -X POST http://localhost:8080/v1/tts \
    -H "Content-Type: application/json" \
    -d '{
      "text": "This is a sample of Airia TTS with fish speech.",
      "reference_id": "glados",
      "format": "wav",
      "top_p": 0.7,
      "temperature": 0.7,
      "seed": 4,
      "repetition_penalty": 1.2
    }' --output sample.wav
```

The `reference_id` parameter can be used to refer to voice reference samples in the references folder. If you do not wish to do cloing of a voice reference, leave out this parameter in API calls. Without a reference voice, be sure to keep the seed the same between calls for consistency in the generated voice.