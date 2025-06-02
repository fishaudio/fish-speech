from __future__ import annotations

import queue
from typing import Any, Callable


class TaskQueue:
    """Minimal in-memory task queue mimicking Celery usage."""

    def __init__(self) -> None:
        self._queue: queue.Queue = queue.Queue()

    def enqueue(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> None:
        self._queue.put((func, args, kwargs))

    def process_next(self) -> None:
        func, args, kwargs = self._queue.get()
        try:
            func(*args, **kwargs)
        finally:
            self._queue.task_done()

    def empty(self) -> bool:
        return self._queue.empty()


__all__ = ["TaskQueue"]
