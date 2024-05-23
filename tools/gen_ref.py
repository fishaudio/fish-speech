from pathlib import Path

import yaml


def scan_folder(base_path):
    wav_lab_pairs = {}

    base = Path(base_path)
    for suf in ["wav", "lab"]:
        for f in base.rglob(f"*.{suf}"):
            relative_path = f.relative_to(base)
            parts = relative_path.parts
            print(parts)
            if len(parts) >= 3:
                character = parts[0]
                emotion = parts[1]

                if character not in wav_lab_pairs:
                    wav_lab_pairs[character] = {}
                if emotion not in wav_lab_pairs[character]:
                    wav_lab_pairs[character][emotion] = []
                wav_lab_pairs[character][emotion].append(str(f.name))

    return wav_lab_pairs


def save_to_yaml(data, output_file):
    with open(output_file, "w", encoding="utf-8") as file:
        yaml.dump(data, file, default_flow_style=False, allow_unicode=True)


base_path = "ref_data"
out_ref_file = "ref_data.yml"

wav_lab_pairs = scan_folder(base_path)
save_to_yaml(wav_lab_pairs, out_ref_file)
