"""Celery configuration file for VoiceReel workers."""

import os

# Broker settings
broker_url = os.getenv("VR_REDIS_URL", "redis://localhost:6379/0")
result_backend = os.getenv("VR_REDIS_URL", "redis://localhost:6379/0")

# Serialization
task_serializer = "json"
result_serializer = "json"
accept_content = ["json"]

# Time zone
timezone = "UTC"
enable_utc = True

# Task execution
task_time_limit = 300  # 5 minutes
task_soft_time_limit = 240  # 4 minutes
task_acks_late = True
worker_prefetch_multiplier = 1

# Task routing
task_routes = {
    "voicereel.tasks.register_speaker": {"queue": "speakers"},
    "voicereel.tasks.synthesize": {"queue": "synthesis"},
    "voicereel.tasks.cleanup_old_files": {"queue": "maintenance"},
}

# Retry policy
task_default_retry_delay = 60
task_max_retries = 3

# Result backend
result_expires = 3600  # 1 hour

# Worker
worker_max_tasks_per_child = 100
worker_disable_rate_limits = False
worker_concurrency = 4  # Adjust based on your hardware

# Beat schedule (for periodic tasks)
beat_schedule = {
    "cleanup-old-files": {
        "task": "voicereel.tasks.cleanup_old_files",
        "schedule": 3600.0,  # Run every hour
        "args": (48,),  # 48 hours max age
    },
}
