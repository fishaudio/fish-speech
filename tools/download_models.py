import os
from huggingface_hub import hf_hub_download

# Donwload
def check_and_download_files(repo_id, file_list, local_dir):
    os.makedirs(local_dir, exist_ok=True)
    for file in file_list:
        file_path = os.path.join(local_dir, file)
        if not os.path.exists(file_path):
            print(f"{file} 不存在，从 Hugging Face 仓库下载...")
            hf_hub_download(
                repo_id=repo_id,
                filename=file,
                resume_download=True,
                local_dir=local_dir,
                local_dir_use_symlinks=False,
            )
        else:
            print(f"{file} 已存在，跳过下载。")

# 1st
repo_id_1 = "fishaudio/fish-speech-1"
local_dir_1 = "./checkpoints"
files_1 = [
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

# 2nd
repo_id_2 = "SpicyqSama007/fish-speech-packed"
local_dir_2 = ".cache/whisper"
files_2 = [
    "medium.pt",
    "small.pt",
]

# 3rd
repo_id_3 = "fishaudio/fish-speech-1"
local_dir_3 = "./"
files_3 = [
    "ffmpeg.exe",
    "ffprobe.exe",
]

# 4rd
repo_id_4 = "SpicyqSama007/fish-speech-packed"
local_dir_4 = "./"
files_4 = [
    "asr-label-win-x64.exe",
]

check_and_download_files(repo_id_1, files_1, local_dir_1)
check_and_download_files(repo_id_2, files_2, local_dir_2)
check_and_download_files(repo_id_3, files_3, local_dir_3)
check_and_download_files(repo_id_4, files_4, local_dir_4)
