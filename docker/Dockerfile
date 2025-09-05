# docker/Dockerfile
# Usage:
#   CUDA:  docker buildx build --build-arg BACKEND=cuda ...
#   Intel: docker buildx build --build-arg BACKEND=intel ...
#   CPU:   docker buildx build --build-arg BACKEND=cpu ...
#   Jetson:docker buildx build --build-arg BACKEND=jetson ...

ARG CUDA_VER=12.9.1
ARG UBUNTU_VER=24.04

ARG PY_VER=3.12
ARG BACKEND=cpu

########################
# Base stage per backend
########################

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


FROM base-${BACKEND} AS builder
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

# COPY docker/requirements.txt /app/requirements.txt


WORKDIR /app

COPY . /app

RUN uv python pin ${PY_VER}
RUN uv sync --extra ${UV_EXTRA}

CMD ["bash"]

