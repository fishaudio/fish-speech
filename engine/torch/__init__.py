"""PyTorch backend: DAC codec + S2-Pro / Drama 3 LLMs + a FastAPI server.

Exports are resolved lazily so importing a single submodule, e.g. Hydra
instantiating a DAC class, does not pull in the whole model stack.
"""

from importlib import import_module

__all__ = ["create_app", "app", "ServerConfig", "TTSRequest", "load_runtime"]

_EXPORTS = {name: ("engine.torch.serve.server", name) for name in __all__}


def __getattr__(name: str):
    try:
        module_name, attr = _EXPORTS[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None

    value = getattr(import_module(module_name), attr)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
