"""Celery tasks for VoiceReel."""

import os
import tempfile
import time
from typing import Dict, List, Any

from celery import Task
from celery.exceptions import SoftTimeLimitExceeded

from .celery_app import app
from .db import init_db
from .caption import export_captions

# Import fish_speech modules (will be integrated later)
# from fish_speech.models.text2semantic.inference import generate_semantic_tokens
# from fish_speech.models.vqgan.inference import synthesis


class DatabaseTask(Task):
    """Base task with database connection."""
    
    _db = None
    
    @property
    def db(self):
        if self._db is None:
            dsn = os.getenv('VR_DSN', ':memory:')
            self._db = init_db(dsn)
        return self._db


@app.task(bind=True, base=DatabaseTask, name='voicereel.tasks.register_speaker')
def register_speaker(self, job_id: str, speaker_id: int, audio_path: str, 
                    script: str, lang: str) -> Dict[str, Any]:
    """Process speaker registration.
    
    Args:
        job_id: Job identifier
        speaker_id: Speaker database ID
        audio_path: Path to reference audio file
        script: Reference script text
        lang: Language code (ISO 639-1)
    
    Returns:
        Dict with processing results
    """
    try:
        # Update job status to processing
        cur = self.db.cursor()
        cur.execute(
            "UPDATE jobs SET status=? WHERE id=?",
            ("processing", job_id)
        )
        self.db.commit()
        
        # TODO: Integrate fish_speech feature extraction
        # For now, simulate processing
        time.sleep(2)
        
        # In production, this would:
        # 1. Load the audio file
        # 2. Extract acoustic features using fish_speech
        # 3. Save speaker embeddings to database
        # 4. Update speaker metadata
        
        # Mark job as succeeded
        cur.execute(
            "UPDATE jobs SET status=? WHERE id=?",
            ("succeeded", job_id)
        )
        self.db.commit()
        
        return {
            'status': 'succeeded',
            'speaker_id': speaker_id,
            'features_extracted': True
        }
        
    except SoftTimeLimitExceeded:
        # Handle timeout
        cur = self.db.cursor()
        cur.execute(
            "UPDATE jobs SET status=? WHERE id=?",
            ("failed", job_id)
        )
        self.db.commit()
        raise
        
    except Exception as e:
        # Handle other errors
        cur = self.db.cursor()
        cur.execute(
            "UPDATE jobs SET status=? WHERE id=?",
            ("failed", job_id)
        )
        self.db.commit()
        
        # Log error details
        self.retry(exc=e, countdown=60)


@app.task(bind=True, base=DatabaseTask, name='voicereel.tasks.synthesize')
def synthesize(self, job_id: str, script: List[Dict[str, str]], 
               output_format: str = 'wav', sample_rate: int = 48000,
               caption_format: str = 'json') -> Dict[str, Any]:
    """Process multi-speaker synthesis.
    
    Args:
        job_id: Job identifier
        script: List of segments with speaker_id and text
        output_format: Audio format (wav/mp3)
        sample_rate: Output sample rate
        caption_format: Caption format (json/vtt/srt)
    
    Returns:
        Dict with synthesis results
    """
    try:
        # Update job status
        cur = self.db.cursor()
        cur.execute(
            "UPDATE jobs SET status=? WHERE id=?",
            ("processing", job_id)
        )
        self.db.commit()
        
        # TODO: Integrate fish_speech synthesis
        # For now, create dummy audio file
        audio_path = os.path.join(tempfile.gettempdir(), f"{job_id}.{output_format}")
        with open(audio_path, 'wb') as f:
            # In production, this would:
            # 1. Load speaker embeddings for each speaker_id
            # 2. Generate semantic tokens for each text segment
            # 3. Synthesize audio using VQGAN
            # 4. Concatenate segments with proper timing
            # 5. Export in requested format
            f.write(b"FAKE_AUDIO_DATA")
        
        # Generate captions with timing
        caption_units = []
        current_time = 0.0
        
        for i, segment in enumerate(script):
            # In production, timing would come from actual synthesis
            duration = len(segment.get('text', '')) * 0.05  # Dummy timing
            caption_units.append({
                'start': current_time,
                'end': current_time + duration,
                'speaker': segment.get('speaker_id'),
                'text': segment.get('text', '')
            })
            current_time += duration + 0.1  # Add small pause
        
        # Export captions
        caption_text = export_captions(caption_units, caption_format)
        caption_path = os.path.join(tempfile.gettempdir(), f"{job_id}.{caption_format}")
        with open(caption_path, 'w', encoding='utf-8') as f:
            f.write(caption_text)
        
        # Update job with results
        cur.execute(
            """UPDATE jobs 
               SET status=?, audio_url=?, caption_path=?, caption_format=? 
               WHERE id=?""",
            ("succeeded", audio_path, caption_path, caption_format, job_id)
        )
        
        # Record usage
        total_duration = current_time
        cur.execute(
            "INSERT INTO usage (ts, length) VALUES (datetime('now'), ?)",
            (total_duration,)
        )
        self.db.commit()
        
        return {
            'status': 'succeeded',
            'audio_path': audio_path,
            'caption_path': caption_path,
            'duration': total_duration
        }
        
    except SoftTimeLimitExceeded:
        cur = self.db.cursor()
        cur.execute(
            "UPDATE jobs SET status=? WHERE id=?",
            ("failed", job_id)
        )
        self.db.commit()
        raise
        
    except Exception as e:
        cur = self.db.cursor()
        cur.execute(
            "UPDATE jobs SET status=? WHERE id=?",
            ("failed", job_id)
        )
        self.db.commit()
        
        self.retry(exc=e, countdown=60)


@app.task(name='voicereel.tasks.cleanup_old_files')
def cleanup_old_files(max_age_hours: float = 48) -> Dict[str, int]:
    """Clean up old audio and caption files.
    
    Args:
        max_age_hours: Maximum age in hours before deletion
        
    Returns:
        Dict with cleanup statistics
    """
    import time
    
    dsn = os.getenv('VR_DSN', ':memory:')
    db = init_db(dsn)
    
    cutoff = time.time() - max_age_hours * 3600
    cur = db.cursor()
    
    # Find old succeeded jobs
    cur.execute(
        "SELECT id, audio_url, caption_path FROM jobs WHERE status='succeeded'"
    )
    
    deleted_files = 0
    deleted_jobs = 0
    
    for job_id, audio_path, caption_path in cur.fetchall():
        keep = False
        
        for path in (audio_path, caption_path):
            if path and os.path.exists(path):
                if os.path.getmtime(path) < cutoff:
                    try:
                        os.remove(path)
                        deleted_files += 1
                    except FileNotFoundError:
                        pass
                else:
                    keep = True
        
        if not keep:
            cur.execute("DELETE FROM jobs WHERE id=?", (job_id,))
            deleted_jobs += 1
    
    db.commit()
    
    return {
        'deleted_files': deleted_files,
        'deleted_jobs': deleted_jobs
    }