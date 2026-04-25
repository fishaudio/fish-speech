"""
conftest.py — pytest session setup for fish-speech audit tests.

Stubs out heavy dependencies (torch, torchaudio, fish_speech sub-packages
other than the two modules under test) so that schema and reference_loader
can be imported and exercised without a GPU or compiled extension.
"""

import importlib.util
import os
import sys
import types

_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _stub_module(name: str, parent: str | None = None, **attrs) -> types.ModuleType:
    mod = types.ModuleType(name)
    mod.__package__ = name
    mod.__path__ = []  # mark as package
    mod.__spec__ = importlib.util.spec_from_loader(name, loader=None)
    for k, v in attrs.items():
        setattr(mod, k, v)
    sys.modules[name] = mod
    if parent and parent in sys.modules:
        setattr(sys.modules[parent], name.split(".")[-1], mod)
    return mod


# ---------------------------------------------------------------------------
# torch / torchaudio stubs
# ---------------------------------------------------------------------------

if "torch" not in sys.modules:
    _stub_module("torch")

if "torchaudio" not in sys.modules:
    ta = _stub_module("torchaudio")
    ta.list_audio_backends = lambda: ["soundfile"]
    ta.load = None  # individual tests patch this
    ta_tf = _stub_module("torchaudio.transforms", parent="torchaudio")
    ta.transforms = ta_tf

if "loguru" not in sys.modules:
    _loguru = _stub_module("loguru")
    _fake_logger = types.SimpleNamespace(
        info=lambda *a, **kw: None,
        warning=lambda *a, **kw: None,
        error=lambda *a, **kw: None,
        debug=lambda *a, **kw: None,
    )
    _loguru.logger = _fake_logger

# ---------------------------------------------------------------------------
# fish_speech package hierarchy — build the skeleton first, then load the
# two real modules (schema + reference_loader) directly from source files.
# ---------------------------------------------------------------------------

# Top-level package
if "fish_speech" not in sys.modules:
    fs = types.ModuleType("fish_speech")
    fs.__package__ = "fish_speech"
    fs.__path__ = [os.path.join(_repo_root, "fish_speech")]
    fs.__spec__ = importlib.util.spec_from_loader("fish_speech", loader=None)
    sys.modules["fish_speech"] = fs

# fish_speech.content_sequence
if "fish_speech.content_sequence" not in sys.modules:
    cs = _stub_module(
        "fish_speech.content_sequence",
        parent="fish_speech",
        TextPart=type("TextPart", (), {}),
        VQPart=type("VQPart", (), {}),
    )

# fish_speech.utils package (stub the __init__ to avoid torch imports)
if "fish_speech.utils" not in sys.modules:
    fu = _stub_module("fish_speech.utils", parent="fish_speech")
    fu.__path__ = [os.path.join(_repo_root, "fish_speech", "utils")]

# fish_speech.utils.file
if "fish_speech.utils.file" not in sys.modules:
    fuf = _stub_module(
        "fish_speech.utils.file",
        parent="fish_speech.utils",
        AUDIO_EXTENSIONS={".wav", ".mp3", ".flac", ".ogg"},
        audio_to_bytes=lambda p: b"",
        list_files=lambda *a, **kw: [],
        read_ref_text=lambda p: "",
    )

# fish_speech.models hierarchy
for _name in ("fish_speech.models", "fish_speech.models.dac", "fish_speech.models.dac.modded_dac"):
    if _name not in sys.modules:
        _stub_module(_name)
sys.modules["fish_speech.models.dac.modded_dac"].DAC = type("DAC", (), {})

# fish_speech.inference_engine package
if "fish_speech.inference_engine" not in sys.modules:
    fie = _stub_module("fish_speech.inference_engine", parent="fish_speech")
    fie.__path__ = [os.path.join(_repo_root, "fish_speech", "inference_engine")]

# ---------------------------------------------------------------------------
# Load fish_speech.utils.schema directly from source (bypasses __init__)
# ---------------------------------------------------------------------------
_schema_file = os.path.join(_repo_root, "fish_speech", "utils", "schema.py")
if "fish_speech.utils.schema" not in sys.modules and os.path.exists(_schema_file):
    _spec = importlib.util.spec_from_file_location("fish_speech.utils.schema", _schema_file)
    _mod = importlib.util.module_from_spec(_spec)
    _mod.__package__ = "fish_speech.utils"
    sys.modules["fish_speech.utils.schema"] = _mod
    _spec.loader.exec_module(_mod)
    sys.modules["fish_speech.utils"].schema = _mod

# ---------------------------------------------------------------------------
# Load fish_speech.inference_engine.reference_loader directly from source
# ---------------------------------------------------------------------------
_rl_file = os.path.join(_repo_root, "fish_speech", "inference_engine", "reference_loader.py")
if "fish_speech.inference_engine.reference_loader" not in sys.modules and os.path.exists(_rl_file):
    _spec = importlib.util.spec_from_file_location(
        "fish_speech.inference_engine.reference_loader", _rl_file
    )
    _mod = importlib.util.module_from_spec(_spec)
    _mod.__package__ = "fish_speech.inference_engine"
    sys.modules["fish_speech.inference_engine.reference_loader"] = _mod
    _spec.loader.exec_module(_mod)
    sys.modules["fish_speech.inference_engine"].reference_loader = _mod
