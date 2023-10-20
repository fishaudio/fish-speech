from pathlib import Path

import librosa
import torch
from torch.utils.data import Dataset
from transformers import WhisperProcessor


class WhisperDataset(Dataset):
    def __init__(self, filelist: str, model_name_or_path: str = "openai/whisper-small"):
        super().__init__()

        self.files = Path(filelist).read_text().splitlines()
        self.processor = WhisperProcessor.from_pretrained(model_name_or_path)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        wav, _ = librosa.load(self.files[idx], sr=16000, mono=True)
        wav = torch.from_numpy(wav).float()
        encoded = self.processor(wav, sampling_rate=16000, return_tensors="pt")

        return {
            "input_values": wav,
            "input_features": encoded.input_features[0],
        }


class WhisperCollator:
    @staticmethod
    def __call__(batch):
        # -> {"input_values": ..., "input_features": ..., "attention_mask": ...}
        max_values_length = max([x["input_values"].shape[-1] for x in batch])

        input_values = []
        input_features = torch.stack([x["input_features"] for x in batch])

        for x in batch:
            values_length = x["input_values"].shape[-1]
            x = torch.nn.functional.pad(x["input_values"], (0, max_values_length - values_length))
            input_values.append(x)

        input_values = torch.stack(input_values)

        return {
            "input_values": input_values,
            "input_features": input_features,
        }


if __name__ == "__main__":
    import soundfile as sf
    from torch.utils.data import DataLoader
    from speech_lm.models.flash_whisper import FlashWhisperForConditionalGeneration

    dataset = WhisperDataset("libritts-r.filelist")
    dataloader = DataLoader(
        dataset, batch_size=1, shuffle=True, collate_fn=WhisperCollator()
    )
    whisper = FlashWhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
    whisper.eval()
    whisper.cuda()

    for batch in dataloader:
        batch = {k: v.cuda() for k, v in batch.items()}
        mask = torch.zeros_like(batch["input_features"])
        # mask[:, :40] = 1

        outputs = whisper.generate(
            inputs=batch["input_features"],
            # attention_mask=batch["attention_mask"],
            max_length=448,
            do_sample=False,
            output_scores=True,
            output_hidden_states=True,
            return_dict_in_generate=True,
            attention_mask=mask,
            # decoder_attention_mask=mask,
        )
        print(outputs.scores[0].shape, outputs.keys(), outputs["sequences"].shape)#, outputs.hidden_states[0].shape)
        print(outputs.encoder_hidden_states[-1][0])

        transcriptions = dataset.processor.batch_decode(outputs["sequences"], skip_special_tokens=True)

        print(transcriptions)
        sf.write("test.wav", batch["input_values"][0].cpu().numpy(), 16000)
        break
