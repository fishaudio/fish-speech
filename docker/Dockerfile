# docker/Dockerfile

# IMPORTANT: The docker images do not contain the checkpoints. You need to mount the checkpoints to the container.

# Build the image:
#   docker build \
#       --platform linux/amd64 \
#       -f docker/Dockerfile \
#       --build-arg BACKEND=[cuda, cpu]
#       --target [webui, server]
#       -t fish-speech:[cuda, cpu] .
#
# e.g. for running the webui:
#   docker build \
#       --platform linux/amd64 \
#       -f docker/Dockerfile \
#       --build-arg BACKEND=cuda \
#       --target webui \
#       -t fish-speech:cuda .
#
# e.g. for running the server:
#   docker build \
#       --platform linux/amd64 \
#       -f docker/Dockerfile \
#       --build-arg BACKEND=cuda \
#       --target server \
#       -t fish-speech:cuda .



# To run the image interactively:
#   docker run \
#       --gpus all \
#       -v /path/to/fish-speech/checkpoints:/app/checkpoints \
#       -e COMPILE=1 \ ... or -e COMPILE=0 \
#       -it fish-speech:cuda




ARG CUDA_VER=12.9.1
ARG UBUNTU_VER=24.04

ARG PY_VER=3.12
ARG BACKEND=cpu

##############################################################
# Base stage per backend
##############################################################

# --- CUDA (x86_64) ---
FROM nvidia/cuda:${CUDA_VER}-cudnn-runtime-ubuntu${UBUNTU_VER} AS base-cuda
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y python3-pip python3-dev git && rm -rf /var/lib/apt/lists/*
# Set correct extra for CUDA
ENV UV_EXTRA=cu129

# --- CPU-only (portable x86_64) ---
FROM python:${PY_VER}-slim AS base-cpu
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*
ENV UV_EXTRA=cpu

##############################################################
# Shared app base stage
##############################################################

FROM base-${BACKEND} AS app-base
ARG PY_VER
ARG BACKEND

ARG DEPENDENCIES="  \
    ca-certificates \
    libsox-dev \
    build-essential \
    cmake \
    libasound-dev \
    portaudio19-dev \
    libportaudio2 \
    libportaudiocpp0 \
    ffmpeg"

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    set -ex \
    && rm -f /etc/apt/apt.conf.d/docker-clean \
    && echo 'Binary::apt::APT::Keep-Downloaded-Packages "true";' >/etc/apt/apt.conf.d/keep-cache \
    && apt-get update \
    && apt-get -y install --no-install-recommends ${DEPENDENCIES} \
    && echo "no" | dpkg-reconfigure dash

# Add uv to the image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app
COPY . /app

# Select correct python version and install dependencies
RUN uv python pin ${PY_VER}
RUN uv sync --extra ${UV_EXTRA}

##############################################################
# App stages
##############################################################

# Gradio WebUI
FROM app-base AS webui

ARG GRADIO_SERVER_NAME="0.0.0.0"
ARG GRADIO_SERVER_PORT=7860
EXPOSE ${GRADIO_SERVER_PORT}

# Write an entrypoint that:
# - adds "--device cpu" if built with BACKEND=cpu
# - adds "--compile" if user passes "compile" at runtime OR COMPILE=1/true
RUN printf '%s\n' '#!/bin/bash' \
'set -e' \
'DEVICE_ARGS=""' \
'if [ "$BACKEND" = "cpu" ]; then DEVICE_ARGS="--device cpu"; fi' \
'EXTRA_ARGS=""' \
'if [ "${1:-}" = "compile" ] || [ "${COMPILE:-}" = "1" ] || [ "${COMPILE:-}" = "true" ]; then EXTRA_ARGS="--compile"; shift || true; fi' \
'exec uv run tools/run_webui.py \
  --llama-checkpoint-path checkpoints/openaudio-s1-mini \
  --decoder-checkpoint-path checkpoints/openaudio-s1-mini/codec.pth \
  --decoder-config-name modded_dac_vq \
  $DEVICE_ARGS $EXTRA_ARGS "$@"' \
> /start_webui.sh && chmod +x /start_webui.sh

ENTRYPOINT ["/start_webui.sh"]


# API Server
FROM app-base AS server
ARG API_SERVER_NAME="0.0.0.0"
ARG API_SERVER_PORT=8080
EXPOSE ${API_SERVER_PORT}

# Write an entrypoint that:
# - adds "--device cpu" if built with BACKEND=cpu
# - adds "--compile" if user passes "compile" at runtime OR COMPILE=1/true
RUN printf '%s\n' '#!/bin/bash' \
'set -e' \
'DEVICE_ARGS=""' \
'if [ "$BACKEND" = "cpu" ]; then DEVICE_ARGS="--device cpu"; fi' \
'EXTRA_ARGS=""' \
'if [ "${1:-}" = "compile" ] || [ "${COMPILE:-}" = "1" ] || [ "${COMPILE:-}" = "true" ]; then EXTRA_ARGS="--compile"; shift || true; fi' \
'exec uv run tools/api_server.py \
  --listen ${API_SERVER_NAME}:${API_SERVER_PORT} \
  --llama-checkpoint-path checkpoints/openaudio-s1-mini \
  --decoder-checkpoint-path checkpoints/openaudio-s1-mini/codec.pth \
  --decoder-config-name modded_dac_vq \
  $DEVICE_ARGS $EXTRA_ARGS "$@"' \
> /start_server.sh && chmod +x /start_server.sh

ENTRYPOINT ["/start_server.sh"]


