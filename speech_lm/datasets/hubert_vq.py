import librosa
from torch.utils.data import Dataset
from pathlib import Path
import torch


class HubertVQDataset(Dataset):
    def __init__(self, filelist):
        super().__init__()

        self.files = Path(filelist).read_text().splitlines()

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        wav, _ = librosa.load(self.files[idx], sr=16000, mono=True)
        wav = torch.from_numpy(wav).float()

        return wav


def collate_fn(batch):
    # -> {"input_values": ..., "attention_mask": ...}
    max_length = max([len(x) for x in batch])

    input_values = []
    attention_mask = []

    for x in batch:
        x_length = len(x)
        x = torch.nn.functional.pad(x, (0, max_length - x_length))
        mask = torch.ones_like(x)
        mask[x_length:] = 0

        input_values.append(x)
        attention_mask.append(mask)

    input_values = torch.stack(input_values)
    attention_mask = torch.stack(attention_mask)

    return {"input_values": input_values, "attention_mask": attention_mask}


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from transformers import HubertForCTC, Wav2Vec2Processor
    import soundfile as sf

    dataset = HubertVQDataset("libritts-r.filelist")
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)
    hubert = HubertForCTC.from_pretrained("facebook/hubert-large-ls960-ft")
    processor = Wav2Vec2Processor.from_pretrained("facebook/hubert-large-ls960-ft")
    hubert.eval()

    for batch in dataloader:
        print(batch)
        logits = hubert(**batch).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = processor.decode(predicted_ids[0])
        print(transcription)

        sf.write("test.wav", batch["input_values"][0].numpy(), 16000)
        break
