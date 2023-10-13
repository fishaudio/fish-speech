from setuptools import find_packages, setup

setup(
    name="speech-lm",
    version="0.0.1",
    packages=find_packages(include=["speech_lm", "speech_lm.*"]),
)
