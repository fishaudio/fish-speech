import tarfile
from pathlib import Path
from tqdm import tqdm
import io
import random
from multiprocessing import Process


def chunked_tarring(rank, file_list, base_folder, output_folder, chunk_size=1024**3):
    chunk_count = 1
    total_size = 0
    saved_count = 0

    buffer = io.BytesIO()
    tar = tarfile.open(fileobj=buffer, mode="w")

    for audio_file in file_list:
        txt_file = audio_file.with_suffix(".txt")
        if not txt_file.exists():
            continue

        file_size = audio_file.stat().st_size + txt_file.stat().st_size
        if total_size + file_size > chunk_size:
            tar.close()

            # write the buffer to disk
            buffer.seek(0)
            with open(output_folder / f"chunk-{rank:03d}-{chunk_count:04d}.tar", "wb") as f:
                f.write(buffer.read())

            chunk_count += 1
            total_size = 0

            buffer.close()
            buffer = io.BytesIO()
            tar = tarfile.open(fileobj=buffer, mode="w")

        tar.add(audio_file, arcname=audio_file.relative_to(base_folder))
        tar.add(txt_file, arcname=txt_file.relative_to(base_folder))

        total_size += file_size

        if saved_count % 1000 == 0:
            print(f"Rank {rank}: {saved_count}/{len(file_list)}")
        
        saved_count += 1

    tar.close()
    buffer.seek(0)
    with open(output_folder / f"chunk-{rank:03d}-{chunk_count:04d}.tar", "wb") as f:
        f.write(buffer.read())
    
    print(f"Rank {rank}: {saved_count}/{len(file_list)}")


if __name__ == "__main__":
    base_folder = Path("/mnt/nvme1/multi-modal-test/WenetSpeech/cleaned")
    output_folder = Path("/mnt/nvme1/multi-modal-test/WenetSpeech/compressed")
    output_folder.mkdir(exist_ok=True, parents=True)
    num_workers = 50

    file_list = list(tqdm(base_folder.rglob("*.flac")))
    random.shuffle(file_list)
    print(f"Total files: {len(file_list)}")

    chunk_size = len(file_list) // num_workers
    processes = []

    for i in range(num_workers):
        start = i * chunk_size
        end = (i + 1) * chunk_size
        if i == num_workers - 1:
            end = len(file_list)

        p = Process(target=chunked_tarring, args=(i, file_list[start:end], base_folder, output_folder))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    print("Done")
