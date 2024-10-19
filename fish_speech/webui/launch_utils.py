import importlib.util
import os
import subprocess
import sys
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import gradio as gr
from gradio.themes.base import Base
from gradio.themes.utils import colors, fonts, sizes

GIT = (
    (Path(os.environ.get("GIT_HOME", "")) / "git").resolve()
    if sys.platform == "win32"
    else "git"
)
GIT = str(GIT)


def is_module_installed(module_name: str) -> bool:
    spec = importlib.util.find_spec(module_name)
    return spec is not None


@lru_cache()
def commit_hash():
    try:
        return subprocess.check_output(
            [GIT, "log", "-1", "--format='%h %s'"], shell=False, encoding="utf8"
        ).strip()
    except Exception:
        return "<none>"


def versions_html():
    import torch

    python_version = ".".join([str(x) for x in sys.version_info[0:3]])
    commit = commit_hash()
    hash = commit.strip("'").split(" ")[0]

    return f"""
version: <a href="https://github.com/fishaudio/fish-speech/commit/{hash}">{hash}</a>
&#x2000;•&#x2000;
python: <span title="{sys.version}">{python_version}</span>
&#x2000;•&#x2000;
torch: {getattr(torch, '__long_version__',torch.__version__)}
&#x2000;•&#x2000;
gradio: {gr.__version__}
&#x2000;•&#x2000;
author: <a href="https://github.com/fishaudio">fishaudio</a>
"""


def version_check(commit):
    try:
        import requests

        commits = requests.get(
            "https://api.github.com/repos/fishaudio/fish-speech/branches/main"
        ).json()
        if commit != "<none>" and commits["commit"]["sha"] != commit:
            print("--------------------------------------------------------")
            print("| You are not up to date with the most recent release. |")
            print("| Consider running `git pull` to update.               |")
            print("--------------------------------------------------------")
        elif commits["commit"]["sha"] == commit:
            print("You are up to date with the most recent release.")
        else:
            print("Not a git clone, can't perform version check.")
    except Exception as e:
        print("version check failed", e)


class Seafoam(Base):
    def __init__(
        self,
        *,
        primary_hue: colors.Color | str = colors.emerald,
        secondary_hue: colors.Color | str = colors.blue,
        neutral_hue: colors.Color | str = colors.blue,
        spacing_size: sizes.Size | str = sizes.spacing_md,
        radius_size: sizes.Size | str = sizes.radius_md,
        text_size: sizes.Size | str = sizes.text_lg,
        font: fonts.Font | str | Iterable[fonts.Font | str] = (
            fonts.GoogleFont("Quicksand"),
            "ui-sans-serif",
            "sans-serif",
        ),
        font_mono: fonts.Font | str | Iterable[fonts.Font | str] = (
            fonts.GoogleFont("IBM Plex Mono"),
            "ui-monospace",
            "monospace",
        ),
    ):
        super().__init__(
            primary_hue=primary_hue,
            secondary_hue=secondary_hue,
            neutral_hue=neutral_hue,
            spacing_size=spacing_size,
            radius_size=radius_size,
            text_size=text_size,
            font=font,
            font_mono=font_mono,
        )
        super().set(
            button_primary_background_fill="linear-gradient(90deg, *primary_300, *secondary_400)",
            button_primary_background_fill_hover="linear-gradient(90deg, *primary_200, *secondary_300)",
            button_primary_text_color="white",
            button_primary_background_fill_dark="linear-gradient(90deg, *primary_600, *secondary_800)",
            slider_color="*secondary_300",
            slider_color_dark="*secondary_600",
            block_title_text_weight="600",
            block_border_width="3px",
            block_shadow="*shadow_drop_lg",
            # button_shadow="*shadow_drop_lg",
            button_small_padding="0px",
            button_large_padding="3px",
        )
