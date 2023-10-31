from setuptools import find_packages, setup

setup(
    name="fish-speech",
    version="0.1.0",
    packages=find_packages(include=["fish_speech", "fish_speech.*"]),
)
