"""Celery application configuration for VoiceReel."""

import os

from celery import Celery

# Initialize Celery app
app = Celery("voicereel")

# Configure Celery
app.conf.update(
    broker_url=os.getenv("VR_REDIS_URL", "redis://localhost:6379/0"),
    result_backend=os.getenv("VR_REDIS_URL", "redis://localhost:6379/0"),
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    # Task routing
    task_routes={
        "voicereel.tasks.register_speaker": {"queue": "speakers"},
        "voicereel.tasks.synthesize": {"queue": "synthesis"},
    },
    # Worker configuration
    worker_prefetch_multiplier=1,
    worker_max_tasks_per_child=100,
    # Task time limits
    task_time_limit=300,  # 5 minutes hard limit
    task_soft_time_limit=240,  # 4 minutes soft limit
    # Retry configuration
    task_default_retry_delay=60,  # 1 minute
    task_max_retries=3,
)

# Auto-discover tasks
app.autodiscover_tasks(["voicereel"])
