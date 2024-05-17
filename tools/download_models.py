import os

from huggingface_hub import hf_hub_download

# 要检查和下载的文件列表
files = [
    "firefly-gan-base-generator.ckpt",
    "README.md",
    "special_tokens_map.json",
    "text2semantic-sft-large-v1.1-4k.pth",
    "text2semantic-sft-medium-v1.1-4k.pth",
    "tokenizer_config.json",
    "tokenizer.json",
    "vits_decoder_v1.1.ckpt",
    "vq-gan-group-fsq-2x1024.pth",
]

# Hugging Face 仓库信息
repo_id = "fishaudio/fish-speech-1"
cache_dir = "./checkpoints"


os.makedirs(cache_dir, exist_ok=True)

# 检查每个文件是否存在，如果不存在则从 Hugging Face 仓库下载
for file in files:
    file_path = os.path.join(cache_dir, file)
    if not os.path.exists(file_path):
        print(f"{file} 不存在，从 Hugging Face 仓库下载...")
        hf_hub_download(
            repo_id=repo_id,
            filename=file,
            cache_dir=cache_dir,
            local_dir_use_symlinks=False,
        )
    else:
        print(f"{file} 已存在，跳过下载。")


files = [
    "medium.pt",
    "small.pt",
]

# Hugging Face 仓库信息
repo_id = "SpicyqSama007/fish-speech-packed"
cache_dir = ".cache/whisper"
os.makedirs(cache_dir, exist_ok=True)

for file in files:
    file_path = os.path.join(cache_dir, file)
    if not os.path.exists(file_path):
        print(f"{file} 不存在，从 Hugging Face 仓库下载...")
        hf_hub_download(
            repo_id=repo_id,
            filename=file,
            cache_dir=cache_dir,
            local_dir_use_symlinks=False,
        )
    else:
        print(f"{file} 已存在，跳过下载。")
