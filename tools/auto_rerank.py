import time
import numpy as np
import torch
import torchaudio
from funasr import AutoModel
from funasr.models.seaco_paraformer.model import SeacoParaformer
from threading import Lock

# Monkey patching to disable hotwords
SeacoParaformer.generate_hotwords_list = lambda self, *args, **kwargs: None


def load_model(*, device="cuda"):
    zh_model = AutoModel(
        model="paraformer-zh",
        device=device,
        disable_pbar=True,
    )
    en_model = AutoModel(
        model="paraformer-en",
        device=device,
        disable_pbar=True,
    )

    return zh_model, en_model


@torch.no_grad()
def batch_asr_internal(model, audios, sr):
    resampled_audios = []
    for audio in audios:
        # 将 NumPy 数组转换为 PyTorch 张量
        if isinstance(audio, np.ndarray):
            audio = torch.from_numpy(audio).float()
        
        # 确保音频是一维的
        if audio.dim() > 1:
            audio = audio.squeeze()
        
        audio = torchaudio.functional.resample(audio, sr, 16000)
        assert audio.dim() == 1
        resampled_audios.append(audio)

    res = model.generate(input=resampled_audios, batch_size=len(resampled_audios))

    results = []
    for r, audio in zip(res, audios):
        text = r["text"]
        duration = len(audio) / sr * 1000
        huge_gap = False

        if "timestamp" in r and len(r["timestamp"]) > 2:
            for timestamp_a, timestamp_b in zip(
                r["timestamp"][:-1], r["timestamp"][1:]
            ):
                # If there is a gap of more than 5 seconds, we consider it as a huge gap
                if timestamp_b[0] - timestamp_a[1] > 5000:
                    huge_gap = True
                    break

            # Doesn't make sense to have a huge gap at the end
            if duration - r["timestamp"][-1][1] > 3000:
                huge_gap = True

        results.append(
            {
                "text": text,
                "duration": duration,
                "huge_gap": huge_gap,
            }
        )

    return results


global_lock = Lock()


def batch_asr(model, audios, sr):
    return batch_asr_internal(model, audios, sr)

def is_chinese(text):
    return True

def calculate_wer(text1,text2):
    words1 = text1.split()
    words2 = text2.split()
    
    # 计算编辑距离
    m, n = len(words1), len(words2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if words1[i-1] == words2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1]) + 1
    
    # 计算WER
    edits = dp[m][n]
    wer = edits / len(words1)
    
    return wer

if __name__ == "__main__":
    zh_model, en_model = load_model()
    audios = [
        torchaudio.load("lengyue.wav")[0][0],
        torchaudio.load("lengyue.wav")[0][0, : 44100 * 5],
    ]
    print(batch_asr(zh_model, audios, 44100))

    start_time = time.time()
    for _ in range(10):
        batch_asr(zh_model, audios, 44100)
    print("Time taken:", time.time() - start_time)
