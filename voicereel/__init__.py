"""VoiceReel package."""

from .caption import export_captions

__all__ = [
    "VoiceReelClient",
    "VoiceReelServer",
    "main",
    "export_captions",
    "create_app",
    "TaskQueue",
    "init_db",
]


def __getattr__(name):
    if name == "VoiceReelClient" or name == "main":
        from .client import VoiceReelClient, main

        globals()["VoiceReelClient"] = VoiceReelClient
        globals()["main"] = main
        return globals()[name]
    if name == "VoiceReelServer":
        from .server import VoiceReelServer

        globals()["VoiceReelServer"] = VoiceReelServer
        return VoiceReelServer
    if name == "create_app":
        from .flask_app import create_app

        globals()["create_app"] = create_app
        return create_app
    if name == "TaskQueue":
        from .task_queue import TaskQueue

        globals()["TaskQueue"] = TaskQueue
        return TaskQueue
    if name == "init_db":
        from .db import init_db

        globals()["init_db"] = init_db
        return init_db
    raise AttributeError(name)
