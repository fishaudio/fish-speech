from pathlib import Path
import subprocess
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import random

def convert_to_flac(src_file_path):
    dst_file_path = src_file_path.with_suffix(".flac")
    dst_file_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        subprocess.check_call(
            ["ffmpeg", "-y", "-i", str(src_file_path), "-acodec", "flac", "-threads", "0", str(dst_file_path)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        # remove the input file
        src_file_path.unlink()
        return True
    except subprocess.CalledProcessError:
        return False


if __name__ == "__main__":
    src_dir = Path("dataset/tts/WenetSpeech/cleaned")

    wav_files = list(src_dir.rglob("*.wav"))
    random.shuffle(wav_files)
    print(f"Found {len(wav_files)} wav files")

    success_counter = 0
    fail_counter = 0

    with Pool(processes=cpu_count(), maxtasksperchild=100) as pool:
        with tqdm(pool.imap_unordered(convert_to_flac, wav_files), total=len(wav_files)) as pbar:
            for success in pbar:
                if success:
                    success_counter += 1
                else:
                    fail_counter += 1
            
            pbar.set_description(f"Success: {success_counter}, Fail: {fail_counter}")

    print(f"Successfully converted: {success_counter}")
    print(f"Failed conversions: {fail_counter}")
