## متطلبات النظام

- ذاكرة GPU: 12GB (للاستنتاج)
- النظام: Linux، WSL

## الإعداد

أولاً تحتاج إلى تثبيت pyaudio و sox، والتي تُستخدم لمعالجة الصوت.

``` bash
apt install portaudio19-dev libsox-dev ffmpeg
```

### Conda

```bash
conda create -n fish-speech python=3.12
conda activate fish-speech

pip install -e .
```

### UV

```bash
uv sync --python 3.12
```

### دعم Intel Arc XPU

```bash
conda create -n fish-speech python=3.12
conda activate fish-speech

conda install libstdcxx -c conda-forge

pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/xpu

pip install -e .
```

!!! warning
    خيار `compile` غير مدعوم على Windows و macOS، إذا كنت تريد التشغيل مع compile، تحتاج إلى تثبيت triton بنفسك.
