# VoiceReel API Dockerfile
FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libpq-dev \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy project files
COPY pyproject.toml README.md ./
COPY fish_speech ./fish_speech
COPY voicereel ./voicereel

# Install Python dependencies
RUN pip install --no-cache-dir -e .
RUN pip install --no-cache-dir \
    'celery[redis]' \
    redis \
    psycopg2-binary \
    gunicorn \
    flower

# Create audio cache directory
RUN mkdir -p /tmp/voicereel_audio

# Expose ports
EXPOSE 8080

# Default command (can be overridden)
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:8080", "voicereel.flask_app:create_app()"]