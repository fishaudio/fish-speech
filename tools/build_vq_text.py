from pathlib import Path

from datasets import Dataset


def parse_data(wav_dir, item):
    text_file = (wav_dir / item["item_name"]).with_suffix(".txt")
    text = text_file.read_text().strip()

    semantic = item["semantic_audio"]
    semantic = [f"<semantic_{x}>" for x in semantic.split(" ")]
    semantic = " ".join(semantic)

    text = f"[INST] {text} [/INST] {semantic} </s>"

    return {
        "text": text,
    }


if __name__ == "__main__":
    # dataset = WenetVQDataset()
    # dataset = list(dataset)
    # print("Initialized dataset.")
    dataset = Dataset.from_csv("data/cn-hubert-wenet-25hz-semantic.tsv", delimiter="\t")
    dataset = dataset.map(
        lambda item: parse_data(Path("data/WenetSpeech"), item), num_proc=64
    )
    dataset = dataset.remove_columns(["item_name", "semantic_audio"])
    dataset = dataset.train_test_split(test_size=0.01)
    print(dataset["test"][0])
    dataset.push_to_hub("fishaudio/wenet-vq", private=True)
