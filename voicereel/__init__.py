"""VoiceReel package."""

from .caption import export_captions

__all__ = ["VoiceReelClient", "VoiceReelServer", "main", "export_captions"]


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
    raise AttributeError(name)
