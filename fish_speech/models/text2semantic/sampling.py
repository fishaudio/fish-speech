"""Sampling defaults for S2-Pro and S3: those of the official Fish Audio SDK
(fish-audio-sdk TTSConfig). The SDK has no top_k; both models use 30."""

DEFAULT_MAX_NEW_TOKENS = 1024  # S3: per chunk
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_P = 0.7
DEFAULT_TOP_K = 30
DEFAULT_REPETITION_PENALTY = 1.2
DEFAULT_CHUNK_LENGTH = 200
