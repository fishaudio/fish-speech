# Speech LLM

## Setup
```bash
# Basic environment setup
conda create -n speech-llm python=3.10
conda activate speech-llm
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

# Install requirements
pip3 install -r requirements.txt

# Install flash-attn
MAX_JOBS=4 pip install flash-attn --no-build-isolation
```
