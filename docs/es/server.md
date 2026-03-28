# Servidor

Esta página cubre la inferencia del lado del servidor para Fish Audio S2, además de enlaces rápidos para la inferencia con WebUI y el despliegue con Docker.

## Inferencia del servidor API

Fish Speech proporciona un punto de entrada de servidor HTTP en `tools/api_server.py`.

### Iniciar el servidor localmente

```bash
python tools/api_server.py \
  --llama-checkpoint-path checkpoints/s2-pro \
  --decoder-checkpoint-path checkpoints/s2-pro/codec.pth \
  --listen 0.0.0.0:8080
```

Opciones comunes:

* `--compile`: habilitar optimización con `torch.compile`
* `--half`: usar modo fp16
* `--api-key`: requerir autenticación mediante bearer token
* `--workers`: establecer la cantidad de procesos worker

### Verificación de estado (health check)

```bash
curl -X GET http://127.0.0.1:8080/v1/health
```

Respuesta esperada:

```json
{"status":"ok"}
```

### Endpoint principal de la API

* `POST /v1/tts` para generación de texto a voz (text-to-speech)
* `POST /v1/vqgan/encode` para codificación VQ
* `POST /v1/vqgan/decode` para decodificación VQ

### Ejemplo de cliente en Python

El modelo base de TTS se selecciona al iniciar el servidor. En el ejemplo anterior, el servidor se inicia con los pesos `checkpoints/s2-pro`, por lo que cada request enviada a `http://127.0.0.1:8080/v1/tts` usará **S2-Pro** automáticamente. No existe un campo `model` por request en `tools/api_client.py` para llamadas al servidor local.

```bash
python tools/api_client.py \
  --url http://127.0.0.1:8080/v1/tts \
  --text "Hello from Fish Speech" \
  --output s2-pro-demo
```

Si quieres seleccionar una voz de referencia guardada, usa `--reference_id`. Esto elige la **referencia de voz**, no el modelo base TTS:

```bash
python tools/api_client.py \
  --url http://127.0.0.1:8080/v1/tts \
  --text "Hello from Fish Speech" \
  --reference_id my-speaker \
  --output s2-pro-demo
```

## Inferencia con WebUI

Para uso con WebUI, ver:

* [WebUI Inference](https://speech.fish.audio/inference/#webui-inference)

## Docker

Para despliegue del servidor o WebUI basado en Docker, ver:

* [Docker Setup](https://speech.fish.audio/install/#docker-setup)

También puedes iniciar directamente el perfil del servidor con Docker Compose:

```bash
docker compose --profile server up
```
