from __future__ import annotations

import os

os.environ["USE_LIBUV"] = "0"
import datetime
import html
import json
import platform
import shutil
import signal
import subprocess
import sys
from pathlib import Path

import gradio as gr
import psutil
import yaml
from loguru import logger
from tqdm import tqdm

PYTHON = os.path.join(os.environ.get("PYTHON_FOLDERPATH", ""), "python")
sys.path.insert(0, "")
print(sys.path)
cur_work_dir = Path(os.getcwd()).resolve()
print("You are in ", str(cur_work_dir))

from fish_speech.i18n import i18n
from fish_speech.webui.launch_utils import Seafoam, is_module_installed, versions_html

config_path = cur_work_dir / "fish_speech" / "configs"
vqgan_yml_path = config_path / "firefly_gan_vq.yaml"
llama_yml_path = config_path / "text2semantic_finetune.yaml"

env = os.environ.copy()
env["no_proxy"] = "127.0.0.1, localhost, 0.0.0.0"

seafoam = Seafoam()


def build_html_error_message(error):
    return f"""
    <div style="color: red; font-weight: bold;">
        {html.escape(error)}
    </div>
    """


def build_html_ok_message(msg):
    return f"""
    <div style="color: green; font-weight: bold;">
        {html.escape(msg)}
    </div>
    """


def build_html_href(link, desc, msg):
    return f"""
    <span style="color: green; font-weight: bold; display: inline-block">
        {html.escape(msg)}
        <a href="{link}">{desc}</a>
    </span>
    """


def load_data_in_raw(path):
    with open(path, "r", encoding="utf-8") as file:
        data = file.read()
    return str(data)


def kill_proc_tree(pid, including_parent=True):
    try:
        parent = psutil.Process(pid)
    except psutil.NoSuchProcess:
        # Process already terminated
        return

    children = parent.children(recursive=True)
    for child in children:
        try:
            os.kill(child.pid, signal.SIGTERM)  # or signal.SIGKILL
        except OSError:
            pass
    if including_parent:
        try:
            os.kill(parent.pid, signal.SIGTERM)  # or signal.SIGKILL
        except OSError:
            pass


system = platform.system()
p_label = None
p_infer = None
p_tensorboard = None


def kill_process(pid):
    if system == "Windows":
        cmd = "taskkill /t /f /pid %s" % pid
        # os.system(cmd)
        subprocess.run(cmd)
    else:
        kill_proc_tree(pid)


def change_label(if_label):
    global p_label
    if if_label == True and p_label is None:
        url = "http://localhost:3000"
        remote_url = "https://text-labeler.pages.dev/"
        try:
            p_label = subprocess.Popen(
                [
                    (
                        "asr-label-linux-x64"
                        if sys.platform == "linux"
                        else "asr-label-win-x64.exe"
                    )
                ]
            )
        except FileNotFoundError:
            logger.warning("asr-label execution not found!")

        yield build_html_href(
            link=remote_url,
            desc=i18n("Optional online ver"),
            msg=i18n("Opened labeler in browser"),
        )

    elif if_label == False and p_label is not None:
        kill_process(p_label.pid)
        p_label = None
        yield build_html_ok_message("Nothing")


def clean_infer_cache():
    import tempfile

    temp_dir = Path(tempfile.gettempdir())
    gradio_dir = str(temp_dir / "gradio")
    try:
        shutil.rmtree(gradio_dir)
        logger.info(f"Deleted cached audios: {gradio_dir}")
    except PermissionError:
        logger.info(f"Permission denied: Unable to delete {gradio_dir}")
    except FileNotFoundError:
        logger.info(f"{gradio_dir} was not found")
    except Exception as e:
        logger.info(f"An error occurred: {e}")


def change_infer(
    if_infer,
    host,
    port,
    infer_decoder_model,
    infer_decoder_config,
    infer_llama_model,
    infer_compile,
):
    global p_infer
    if if_infer == True and p_infer == None:
        env = os.environ.copy()

        env["GRADIO_SERVER_NAME"] = host
        env["GRADIO_SERVER_PORT"] = port
        # 启动第二个进程
        url = f"http://{host}:{port}"
        yield build_html_ok_message(
            i18n("Inferring interface is launched at {}").format(url)
        )

        clean_infer_cache()

        p_infer = subprocess.Popen(
            [
                PYTHON,
                "tools/webui.py",
                "--decoder-checkpoint-path",
                infer_decoder_model,
                "--decoder-config-name",
                infer_decoder_config,
                "--llama-checkpoint-path",
                infer_llama_model,
            ]
            + (["--compile"] if infer_compile == "Yes" else []),
            env=env,
        )

    elif if_infer == False and p_infer is not None:
        kill_process(p_infer.pid)
        p_infer = None
        yield build_html_error_message(i18n("Infer interface is closed"))


js = load_data_in_raw("fish_speech/webui/js/animate.js")
css = load_data_in_raw("fish_speech/webui/css/style.css")

data_pre_output = (cur_work_dir / "data").resolve()
default_model_output = (cur_work_dir / "results").resolve()
default_filelist = data_pre_output / "detect.list"
data_pre_output.mkdir(parents=True, exist_ok=True)

items = []
dict_items = {}


def load_yaml_data_in_fact(yml_path):
    with open(yml_path, "r", encoding="utf-8") as file:
        yml = yaml.safe_load(file)
    return yml


def write_yaml_data_in_fact(yml, yml_path):
    with open(yml_path, "w", encoding="utf-8") as file:
        yaml.safe_dump(yml, file, allow_unicode=True)
    return yml


def generate_tree(directory, depth=0, max_depth=None, prefix=""):
    if max_depth is not None and depth > max_depth:
        return ""

    tree_str = ""
    files = []
    directories = []
    for item in os.listdir(directory):
        if os.path.isdir(os.path.join(directory, item)):
            directories.append(item)
        else:
            files.append(item)

    entries = directories + files
    for i, entry in enumerate(entries):
        connector = "├── " if i < len(entries) - 1 else "└── "
        tree_str += f"{prefix}{connector}{entry}<br />"
        if i < len(directories):
            extension = "│   " if i < len(entries) - 1 else "    "
            tree_str += generate_tree(
                os.path.join(directory, entry),
                depth + 1,
                max_depth,
                prefix=prefix + extension,
            )
    return tree_str


def new_explorer(data_path, max_depth):
    return gr.Markdown(
        elem_classes=["scrollable-component"],
        value=generate_tree(data_path, max_depth=max_depth),
    )


def add_item(
    folder: str,
    method: str,
    label_lang: str,
    if_initial_prompt: bool,
    initial_prompt: str | None,
):
    folder = folder.strip(" ").strip('"')

    folder_path = Path(folder)

    if folder and folder not in items and data_pre_output not in folder_path.parents:
        if folder_path.is_dir():
            items.append(folder)
            dict_items[folder] = dict(
                type="folder",
                method=method,
                label_lang=label_lang,
                initial_prompt=initial_prompt if if_initial_prompt else None,
            )
        elif folder:
            err = folder
            return gr.Checkboxgroup(choices=items), build_html_error_message(
                i18n("Invalid path: {}").format(err)
            )

    formatted_data = json.dumps(dict_items, ensure_ascii=False, indent=4)
    logger.info("After Adding: " + formatted_data)
    gr.Info(formatted_data)
    return gr.Checkboxgroup(choices=items), build_html_ok_message(
        i18n("Added path successfully!")
    )


def remove_items(selected_items):
    global items, dict_items
    to_remove = [item for item in items if item in selected_items]
    for item in to_remove:
        del dict_items[item]
    items = [item for item in items if item in dict_items.keys()]
    formatted_data = json.dumps(dict_items, ensure_ascii=False, indent=4)
    logger.info(formatted_data)
    gr.Warning("After Removing: " + formatted_data)
    return gr.Checkboxgroup(choices=items, value=[]), build_html_ok_message(
        i18n("Removed path successfully!")
    )


def show_selected(options):
    selected_options = ", ".join(options)

    if options:
        return i18n("Selected: {}").format(selected_options)
    else:
        return i18n("No selected options")


from pydub import AudioSegment


def convert_to_mono_in_place(audio_path: Path):
    audio = AudioSegment.from_file(audio_path)
    if audio.channels > 1:
        mono_audio = audio.set_channels(1)
        mono_audio.export(audio_path, format=audio_path.suffix[1:])
        logger.info(f"Convert {audio_path} successfully")


def list_copy(list_file_path, method):
    wav_root = data_pre_output
    lst = []
    with list_file_path.open("r", encoding="utf-8") as file:
        for line in tqdm(file, desc="Processing audio/transcript"):
            wav_path, speaker_name, language, text = line.strip().split("|")
            original_wav_path = Path(wav_path)
            target_wav_path = (
                wav_root / original_wav_path.parent.name / original_wav_path.name
            )
            lst.append(f"{target_wav_path}|{speaker_name}|{language}|{text}")
            if target_wav_path.is_file():
                continue
            target_wav_path.parent.mkdir(parents=True, exist_ok=True)
            if method == i18n("Copy"):
                shutil.copy(original_wav_path, target_wav_path)
            else:
                shutil.move(original_wav_path, target_wav_path.parent)
            convert_to_mono_in_place(target_wav_path)
            original_lab_path = original_wav_path.with_suffix(".lab")
            target_lab_path = (
                wav_root
                / original_wav_path.parent.name
                / original_wav_path.with_suffix(".lab").name
            )
            if target_lab_path.is_file():
                continue
            if method == i18n("Copy"):
                shutil.copy(original_lab_path, target_lab_path)
            else:
                shutil.move(original_lab_path, target_lab_path.parent)

    if method == i18n("Move"):
        with list_file_path.open("w", encoding="utf-8") as file:
            file.writelines("\n".join(lst))

    del lst
    return build_html_ok_message(i18n("Use filelist"))


def check_files(data_path: str, max_depth: int, label_model: str, label_device: str):
    global dict_items
    data_path = Path(data_path)
    gr.Warning("Pre-processing begins...")
    for item, content in dict_items.items():
        item_path = Path(item)
        tar_path = data_path / item_path.name

        if content["type"] == "folder" and item_path.is_dir():
            if content["method"] == i18n("Copy"):
                os.makedirs(tar_path, exist_ok=True)
                shutil.copytree(
                    src=str(item_path), dst=str(tar_path), dirs_exist_ok=True
                )
            elif not tar_path.is_dir():
                shutil.move(src=str(item_path), dst=str(tar_path))

            for suf in ["wav", "flac", "mp3"]:
                for audio_path in tar_path.glob(f"**/*.{suf}"):
                    convert_to_mono_in_place(audio_path)

            cur_lang = content["label_lang"]
            initial_prompt = content["initial_prompt"]

            transcribe_cmd = [
                PYTHON,
                "tools/whisper_asr.py",
                "--model-size",
                label_model,
                "--device",
                label_device,
                "--audio-dir",
                tar_path,
                "--save-dir",
                tar_path,
                "--language",
                cur_lang,
            ]

            if initial_prompt is not None:
                transcribe_cmd += ["--initial-prompt", initial_prompt]

            if cur_lang != "IGNORE":
                try:
                    gr.Warning("Begin To Transcribe")
                    subprocess.run(
                        transcribe_cmd,
                        env=env,
                    )
                except Exception:
                    print("Transcription error occurred")

        elif content["type"] == "file" and item_path.is_file():
            list_copy(item_path, content["method"])

    return build_html_ok_message(i18n("Move files successfully")), new_explorer(
        data_path, max_depth=max_depth
    )


def generate_folder_name():
    now = datetime.datetime.now()
    folder_name = now.strftime("%Y%m%d_%H%M%S")
    return folder_name


def train_process(
    data_path: str,
    option: str,
    # llama config
    llama_ckpt,
    llama_base_config,
    llama_lr,
    llama_maxsteps,
    llama_data_num_workers,
    llama_data_batch_size,
    llama_data_max_length,
    llama_precision,
    llama_check_interval,
    llama_grad_batches,
    llama_use_speaker,
    llama_use_lora,
):

    backend = "nccl" if sys.platform == "linux" else "gloo"

    new_project = generate_folder_name()
    print("New Project Name: ", new_project)

    if option == "VQGAN":
        msg = "Skipped VQGAN Training."
        gr.Warning(msg)
        logger.info(msg)

    if option == "LLAMA":
        msg = "LLAMA Training begins..."
        gr.Warning(msg)
        logger.info(msg)
        subprocess.run(
            [
                PYTHON,
                "tools/vqgan/extract_vq.py",
                str(data_pre_output),
                "--num-workers",
                "1",
                "--batch-size",
                "16",
                "--config-name",
                "firefly_gan_vq",
                "--checkpoint-path",
                "checkpoints/fish-speech-1.4/firefly-gan-vq-fsq-8x1024-21hz-generator.pth",
            ]
        )

        subprocess.run(
            [
                PYTHON,
                "tools/llama/build_dataset.py",
                "--input",
                str(data_pre_output),
                "--text-extension",
                ".lab",
                "--num-workers",
                "16",
            ]
        )
        ckpt_path = "checkpoints/fish-speech-1.4/model.pth"
        lora_prefix = "lora_" if llama_use_lora else ""
        llama_name = lora_prefix + "text2semantic_" + new_project
        latest = next(
            iter(
                sorted(
                    [
                        str(p.relative_to("results"))
                        for p in Path("results").glob(lora_prefix + "text2sem*/")
                    ],
                    reverse=True,
                )
            ),
            llama_name,
        )
        project = (
            llama_name
            if llama_ckpt == i18n("new")
            else (
                latest
                if llama_ckpt == i18n("latest")
                else Path(llama_ckpt).relative_to("results")
            )
        )
        logger.info(project)

        if llama_check_interval > llama_maxsteps:
            llama_check_interval = llama_maxsteps

        train_cmd = [
            PYTHON,
            "fish_speech/train.py",
            "--config-name",
            "text2semantic_finetune",
            f"project={project}",
            f"trainer.strategy.process_group_backend={backend}",
            f"train_dataset.proto_files={str(['data/quantized-dataset-ft'])}",
            f"val_dataset.proto_files={str(['data/quantized-dataset-ft'])}",
            f"model.optimizer.lr={llama_lr}",
            f"trainer.max_steps={llama_maxsteps}",
            f"data.num_workers={llama_data_num_workers}",
            f"data.batch_size={llama_data_batch_size}",
            f"max_length={llama_data_max_length}",
            f"trainer.precision={llama_precision}",
            f"trainer.val_check_interval={llama_check_interval}",
            f"trainer.accumulate_grad_batches={llama_grad_batches}",
            f"train_dataset.interactive_prob={llama_use_speaker}",
        ] + ([f"+lora@model.model.lora_config=r_8_alpha_16"] if llama_use_lora else [])
        logger.info(train_cmd)
        subprocess.run(train_cmd)

    return build_html_ok_message(i18n("Training stopped"))


def tensorboard_process(
    if_tensorboard: bool,
    tensorboard_dir: str,
    host: str,
    port: str,
):
    global p_tensorboard
    if if_tensorboard == True and p_tensorboard == None:
        url = f"http://{host}:{port}"
        yield build_html_ok_message(
            i18n("Tensorboard interface is launched at {}").format(url)
        )
        prefix = ["tensorboard"]
        if Path("fishenv").exists():
            prefix = ["fishenv/env/python.exe", "fishenv/env/Scripts/tensorboard.exe"]

        p_tensorboard = subprocess.Popen(
            prefix
            + [
                "--logdir",
                tensorboard_dir,
                "--host",
                host,
                "--port",
                port,
                "--reload_interval",
                "120",
            ]
        )
    elif if_tensorboard == False and p_tensorboard != None:
        kill_process(p_tensorboard.pid)
        p_tensorboard = None
        yield build_html_error_message(i18n("Tensorboard interface is closed"))


def fresh_tb_dir():
    return gr.Dropdown(
        choices=[str(p) for p in Path("results").glob("**/tensorboard/")]
    )


def list_decoder_models():
    paths = [str(p) for p in Path("checkpoints").glob("fish*/firefly*.pth")]
    if not paths:
        logger.warning("No decoder model found")
    return paths


def list_llama_models():
    choices = [str(p.parent) for p in Path("checkpoints").glob("merged*/*model*.pth")]
    choices += [str(p.parent) for p in Path("checkpoints").glob("fish*/*model*.pth")]
    choices += [str(p.parent) for p in Path("checkpoints").glob("fs*/*model*.pth")]
    choices = sorted(choices, reverse=True)
    if not choices:
        logger.warning("No LLaMA model found")
    return choices


def list_lora_llama_models():
    choices = sorted(
        [str(p) for p in Path("results").glob("lora*/**/*.ckpt")], reverse=True
    )
    if not choices:
        logger.warning("No LoRA LLaMA model found")
    return choices


def fresh_decoder_model():
    return gr.Dropdown(choices=list_decoder_models())


def fresh_llama_ckpt(llama_use_lora):
    return gr.Dropdown(
        choices=[i18n("latest"), i18n("new")]
        + (
            [str(p) for p in Path("results").glob("text2sem*/")]
            if not llama_use_lora
            else [str(p) for p in Path("results").glob("lora_*/")]
        )
    )


def fresh_llama_model():
    return gr.Dropdown(choices=list_llama_models())


def llama_lora_merge(llama_weight, lora_llama_config, lora_weight, llama_lora_output):
    if (
        lora_weight is None
        or not Path(lora_weight).exists()
        or not Path(llama_weight).exists()
    ):
        return build_html_error_message(
            i18n(
                "Path error, please check the model file exists in the corresponding path"
            )
        )
    gr.Warning("Merging begins...")
    merge_cmd = [
        PYTHON,
        "tools/llama/merge_lora.py",
        "--lora-config",
        "r_8_alpha_16",
        "--lora-weight",
        lora_weight,
        "--output",
        llama_lora_output + "_" + generate_folder_name(),
    ]
    logger.info(merge_cmd)
    subprocess.run(merge_cmd)
    return build_html_ok_message(i18n("Merge successfully"))


def llama_quantify(llama_weight, quantify_mode):
    if llama_weight is None or not Path(llama_weight).exists():
        return build_html_error_message(
            i18n(
                "Path error, please check the model file exists in the corresponding path"
            )
        )

    gr.Warning("Quantifying begins...")

    now = generate_folder_name()
    quantify_cmd = [
        PYTHON,
        "tools/llama/quantize.py",
        "--checkpoint-path",
        llama_weight,
        "--mode",
        quantify_mode,
        "--timestamp",
        now,
    ]
    logger.info(quantify_cmd)
    subprocess.run(quantify_cmd)
    if quantify_mode == "int8":
        quantize_path = str(
            Path(os.getcwd()) / "checkpoints" / f"fs-1.2-{quantify_mode}-{now}"
        )
    else:
        quantize_path = str(
            Path(os.getcwd()) / "checkpoints" / f"fs-1.2-{quantify_mode}-g128-{now}"
        )
    return build_html_ok_message(
        i18n("Quantify successfully") + f"Path: {quantize_path}"
    )


init_vqgan_yml = load_yaml_data_in_fact(vqgan_yml_path)
init_llama_yml = load_yaml_data_in_fact(llama_yml_path)

with gr.Blocks(
    head="<style>\n" + css + "\n</style>",
    js=js,
    theme=seafoam,
    analytics_enabled=False,
    title="Fish Speech",
) as demo:
    with gr.Row():
        with gr.Column():
            with gr.Tab("\U0001F4D6 " + i18n("Data Preprocessing")):
                with gr.Row():
                    textbox = gr.Textbox(
                        label="\U0000270F "
                        + i18n("Input Audio & Source Path for Transcription"),
                        info=i18n("Speaker is identified by the folder name"),
                        interactive=True,
                    )
                with gr.Row(equal_height=False):
                    with gr.Column():
                        output_radio = gr.Radio(
                            label="\U0001F4C1 "
                            + i18n("Select source file processing method"),
                            choices=[i18n("Copy"), i18n("Move")],
                            value=i18n("Copy"),
                            interactive=True,
                        )
                    with gr.Column():
                        error = gr.HTML(label=i18n("Error Message"))
                        if_label = gr.Checkbox(
                            label=i18n("Open Labeler WebUI"), scale=0, show_label=True
                        )

                with gr.Row():
                    label_device = gr.Dropdown(
                        label=i18n("Labeling Device"),
                        info=i18n(
                            "It is recommended to use CUDA, if you have low configuration, use CPU"
                        ),
                        choices=["cpu", "cuda"],
                        value="cuda",
                        interactive=True,
                    )
                    label_model = gr.Dropdown(
                        label=i18n("Whisper Model"),
                        info=i18n("Faster Whisper, Up to 5g GPU memory usage"),
                        choices=["large-v3", "medium"],
                        value="large-v3",
                        interactive=True,
                    )
                    label_radio = gr.Dropdown(
                        label=i18n("Optional Label Language"),
                        info=i18n(
                            "If there is no corresponding text for the audio, apply ASR for assistance, support .txt or .lab format"
                        ),
                        choices=[
                            (i18n("Chinese"), "zh"),
                            (i18n("English"), "en"),
                            (i18n("Japanese"), "ja"),
                            (i18n("Disabled"), "IGNORE"),
                            (i18n("auto"), "auto"),
                        ],
                        value="IGNORE",
                        interactive=True,
                    )

                with gr.Row():
                    if_initial_prompt = gr.Checkbox(
                        value=False,
                        label=i18n("Enable Initial Prompt"),
                        min_width=120,
                        scale=0,
                    )
                    initial_prompt = gr.Textbox(
                        label=i18n("Initial Prompt"),
                        info=i18n(
                            "Initial prompt can provide contextual or vocabulary-specific guidance to the model."
                        ),
                        placeholder="This audio introduces the basic concepts and applications of artificial intelligence and machine learning.",
                        interactive=False,
                    )

                with gr.Row():
                    add_button = gr.Button(
                        "\U000027A1 " + i18n("Add to Processing Area"),
                        variant="primary",
                    )
                    remove_button = gr.Button(
                        "\U000026D4 " + i18n("Remove Selected Data")
                    )

            with gr.Tab("\U0001F6E0 " + i18n("Training Configuration")):
                with gr.Row():
                    model_type_radio = gr.Radio(
                        label=i18n(
                            "Select the model to be trained (Depending on the Tab page you are on)"
                        ),
                        interactive=False,
                        choices=["VQGAN", "LLAMA"],
                        value="VQGAN",
                    )
                with gr.Row():
                    with gr.Tabs():
                        with gr.Tab(label=i18n("VQGAN Configuration")) as vqgan_page:
                            gr.HTML("You don't need to train this model!")

                        with gr.Tab(label=i18n("LLAMA Configuration")) as llama_page:
                            with gr.Row(equal_height=False):
                                llama_use_lora = gr.Checkbox(
                                    label=i18n("Use LoRA"),
                                    info=i18n(
                                        "Use LoRA can save GPU memory, but may reduce the quality of the model"
                                    ),
                                    value=True,
                                    interactive=True,
                                )
                                llama_ckpt = gr.Dropdown(
                                    label=i18n("Select LLAMA ckpt"),
                                    choices=[i18n("latest"), i18n("new")]
                                    + [
                                        str(p)
                                        for p in Path("results").glob("text2sem*/")
                                    ]
                                    + [str(p) for p in Path("results").glob("lora*/")],
                                    value=i18n("latest"),
                                    interactive=True,
                                )
                            with gr.Row(equal_height=False):
                                llama_lr_slider = gr.Slider(
                                    label=i18n("Initial Learning Rate"),
                                    info=i18n(
                                        "lr smaller -> usually train slower but more stable"
                                    ),
                                    interactive=True,
                                    minimum=1e-5,
                                    maximum=1e-4,
                                    step=1e-5,
                                    value=5e-5,
                                )
                                llama_maxsteps_slider = gr.Slider(
                                    label=i18n("Maximum Training Steps"),
                                    info=i18n(
                                        "recommend: max_steps = num_audios // batch_size * (2 to 5)"
                                    ),
                                    interactive=True,
                                    minimum=1,
                                    maximum=10000,
                                    step=1,
                                    value=50,
                                )
                            with gr.Row(equal_height=False):
                                llama_base_config = gr.Dropdown(
                                    label=i18n("Model Size"),
                                    choices=[
                                        "text2semantic_finetune",
                                    ],
                                    value="text2semantic_finetune",
                                )
                                llama_data_num_workers_slider = gr.Slider(
                                    label=i18n("Number of Workers"),
                                    minimum=1,
                                    maximum=32,
                                    step=1,
                                    value=4,
                                )
                            with gr.Row(equal_height=False):
                                llama_data_batch_size_slider = gr.Slider(
                                    label=i18n("Batch Size"),
                                    interactive=True,
                                    minimum=1,
                                    maximum=32,
                                    step=1,
                                    value=2,
                                )
                                llama_data_max_length_slider = gr.Slider(
                                    label=i18n("Maximum Length per Sample"),
                                    interactive=True,
                                    minimum=1024,
                                    maximum=4096,
                                    step=128,
                                    value=2048,
                                )
                            with gr.Row(equal_height=False):
                                llama_precision_dropdown = gr.Dropdown(
                                    label=i18n("Precision"),
                                    info=i18n(
                                        "bf16-true is recommended for 30+ series GPU, 16-mixed is recommended for 10+ series GPU"
                                    ),
                                    interactive=True,
                                    choices=["32", "bf16-true", "16-mixed"],
                                    value="bf16-true",
                                )
                                llama_check_interval_slider = gr.Slider(
                                    label=i18n("Save model every n steps"),
                                    info=i18n(
                                        "make sure that it's not greater than max_steps"
                                    ),
                                    interactive=True,
                                    minimum=1,
                                    maximum=1000,
                                    step=1,
                                    value=50,
                                )
                            with gr.Row(equal_height=False):
                                llama_grad_batches = gr.Slider(
                                    label=i18n("Accumulate Gradient Batches"),
                                    interactive=True,
                                    minimum=1,
                                    maximum=20,
                                    step=1,
                                    value=init_llama_yml["trainer"][
                                        "accumulate_grad_batches"
                                    ],
                                )
                                llama_use_speaker = gr.Slider(
                                    label=i18n(
                                        "Probability of applying Speaker Condition"
                                    ),
                                    interactive=True,
                                    minimum=0.1,
                                    maximum=1.0,
                                    step=0.05,
                                    value=init_llama_yml["train_dataset"][
                                        "interactive_prob"
                                    ],
                                )

                        with gr.Tab(label=i18n("Merge LoRA"), id=4):
                            with gr.Row(equal_height=False):
                                llama_weight = gr.Dropdown(
                                    label=i18n("Base LLAMA Model"),
                                    info=i18n(
                                        "Type the path or select from the dropdown"
                                    ),
                                    choices=[
                                        "checkpoints/fish-speech-1.4/model.pth",
                                    ],
                                    value="checkpoints/fish-speech-1.4/model.pth",
                                    allow_custom_value=True,
                                    interactive=True,
                                )
                            with gr.Row(equal_height=False):
                                lora_weight = gr.Dropdown(
                                    label=i18n("LoRA Model to be merged"),
                                    info=i18n(
                                        "Type the path or select from the dropdown"
                                    ),
                                    choices=[
                                        str(p)
                                        for p in Path("results").glob("lora*/**/*.ckpt")
                                    ],
                                    allow_custom_value=True,
                                    interactive=True,
                                )
                                lora_llama_config = gr.Dropdown(
                                    label=i18n("LLAMA Model Config"),
                                    info=i18n(
                                        "Type the path or select from the dropdown"
                                    ),
                                    choices=[
                                        "text2semantic_finetune",
                                    ],
                                    value="text2semantic_finetune",
                                    allow_custom_value=True,
                                )
                            with gr.Row(equal_height=False):
                                llama_lora_output = gr.Dropdown(
                                    label=i18n("Output Path"),
                                    info=i18n(
                                        "Type the path or select from the dropdown"
                                    ),
                                    value="checkpoints/merged",
                                    choices=["checkpoints/merged"],
                                    allow_custom_value=True,
                                    interactive=True,
                                )
                            with gr.Row(equal_height=False):
                                llama_lora_merge_btn = gr.Button(
                                    value=i18n("Merge"), variant="primary"
                                )

                        with gr.Tab(label=i18n("Model Quantization"), id=5):
                            with gr.Row(equal_height=False):
                                llama_weight_to_quantify = gr.Dropdown(
                                    label=i18n("Base LLAMA Model"),
                                    info=i18n(
                                        "Type the path or select from the dropdown"
                                    ),
                                    choices=list_llama_models(),
                                    value="checkpoints/fish-speech-1.4",
                                    allow_custom_value=True,
                                    interactive=True,
                                )
                                quantify_mode = gr.Dropdown(
                                    label=i18n("Post-quantification Precision"),
                                    info=i18n(
                                        "The lower the quantitative precision, the more the effectiveness may decrease, but the greater the efficiency will increase"
                                    ),
                                    choices=["int8", "int4"],
                                    value="int8",
                                    allow_custom_value=False,
                                    interactive=True,
                                )
                            with gr.Row(equal_height=False):
                                llama_quantify_btn = gr.Button(
                                    value=i18n("Quantify"), variant="primary"
                                )

                        with gr.Tab(label="Tensorboard", id=6):
                            with gr.Row(equal_height=False):
                                tb_host = gr.Textbox(
                                    label=i18n("Tensorboard Host"), value="127.0.0.1"
                                )
                                tb_port = gr.Textbox(
                                    label=i18n("Tensorboard Port"), value="11451"
                                )
                            with gr.Row(equal_height=False):
                                tb_dir = gr.Dropdown(
                                    label=i18n("Tensorboard Log Path"),
                                    allow_custom_value=True,
                                    choices=[
                                        str(p)
                                        for p in Path("results").glob("**/tensorboard/")
                                    ],
                                )
                            with gr.Row(equal_height=False):
                                if_tb = gr.Checkbox(
                                    label=i18n("Open Tensorboard"),
                                )

            with gr.Tab("\U0001F9E0 " + i18n("Inference Configuration")):
                with gr.Column():
                    with gr.Row():
                        with gr.Accordion(
                            label="\U0001F5A5 "
                            + i18n("Inference Server Configuration"),
                            open=False,
                        ):
                            with gr.Row():
                                infer_host_textbox = gr.Textbox(
                                    label=i18n("WebUI Host"), value="127.0.0.1"
                                )
                                infer_port_textbox = gr.Textbox(
                                    label=i18n("WebUI Port"), value="7862"
                                )
                            with gr.Row():
                                infer_decoder_model = gr.Dropdown(
                                    label=i18n("Decoder Model Path"),
                                    info=i18n(
                                        "Type the path or select from the dropdown"
                                    ),
                                    choices=list_decoder_models(),
                                    value="checkpoints/fish-speech-1.4/firefly-gan-vq-fsq-8x1024-21hz-generator.pth",
                                    allow_custom_value=True,
                                )
                                infer_decoder_config = gr.Dropdown(
                                    label=i18n("Decoder Model Config"),
                                    info=i18n("Changing with the Model Path"),
                                    value="firefly_gan_vq",
                                    choices=[
                                        "firefly_gan_vq",
                                    ],
                                    allow_custom_value=True,
                                )
                            with gr.Row():
                                infer_llama_model = gr.Dropdown(
                                    label=i18n("LLAMA Model Path"),
                                    info=i18n(
                                        "Type the path or select from the dropdown"
                                    ),
                                    value="checkpoints/fish-speech-1.4",
                                    choices=list_llama_models(),
                                    allow_custom_value=True,
                                )

                            with gr.Row():
                                infer_compile = gr.Radio(
                                    label=i18n("Compile Model"),
                                    info=i18n(
                                        "Compile the model can significantly reduce the inference time, but will increase cold start time"
                                    ),
                                    choices=["Yes", "No"],
                                    value=(
                                        "Yes" if (sys.platform == "linux") else "No"
                                    ),
                                    interactive=is_module_installed("triton"),
                                )

                    with gr.Row():
                        infer_checkbox = gr.Checkbox(
                            label=i18n("Open Inference Server")
                        )
                        infer_error = gr.HTML(label=i18n("Inference Server Error"))

        with gr.Column():
            train_error = gr.HTML(label=i18n("Training Error"))
            checkbox_group = gr.CheckboxGroup(
                label="\U0001F4CA " + i18n("Data Source"),
                info=i18n(
                    "The path of the input folder on the left or the filelist. Whether checked or not, it will be used for subsequent training in this list."
                ),
                elem_classes=["data_src"],
            )
            train_box = gr.Textbox(
                label=i18n("Data Preprocessing Path"),
                value=str(data_pre_output),
                interactive=False,
            )
            model_box = gr.Textbox(
                label="\U0001F4BE " + i18n("Model Output Path"),
                value=str(default_model_output),
                interactive=False,
            )

            with gr.Accordion(
                i18n(
                    "View the status of the preprocessing folder (use the slider to control the depth of the tree)"
                ),
                elem_classes=["scrollable-component"],
                elem_id="file_accordion",
            ):
                tree_slider = gr.Slider(
                    minimum=0,
                    maximum=3,
                    value=0,
                    step=1,
                    show_label=False,
                    container=False,
                )
                file_markdown = new_explorer(str(data_pre_output), 0)
            with gr.Row(equal_height=False):
                admit_btn = gr.Button(
                    "\U00002705 " + i18n("File Preprocessing"),
                    variant="primary",
                )
                fresh_btn = gr.Button("\U0001F503", scale=0, min_width=80)
                help_button = gr.Button("\U00002753", scale=0, min_width=80)  # question
                train_btn = gr.Button(i18n("Start Training"), variant="primary")

    footer = load_data_in_raw("fish_speech/webui/html/footer.html")
    footer = footer.format(
        versions=versions_html(),
        api_docs="https://speech.fish.audio/inference/#http-api",
    )
    gr.HTML(footer, elem_id="footer")
    vqgan_page.select(lambda: "VQGAN", None, model_type_radio)
    llama_page.select(lambda: "LLAMA", None, model_type_radio)
    add_button.click(
        fn=add_item,
        inputs=[textbox, output_radio, label_radio, if_initial_prompt, initial_prompt],
        outputs=[checkbox_group, error],
    )
    remove_button.click(
        fn=remove_items, inputs=[checkbox_group], outputs=[checkbox_group, error]
    )
    checkbox_group.change(fn=show_selected, inputs=checkbox_group, outputs=[error])
    help_button.click(
        fn=None,
        js='() => { window.open("https://speech.fish.audio/", "newwindow", "height=100, width=400, '
        'toolbar=no, menubar=no, scrollbars=no, resizable=no, location=no, status=no")}',
    )
    if_label.change(fn=change_label, inputs=[if_label], outputs=[error])
    if_initial_prompt.change(
        fn=lambda x: gr.Textbox(value="", interactive=x),
        inputs=[if_initial_prompt],
        outputs=[initial_prompt],
    )
    train_btn.click(
        fn=train_process,
        inputs=[
            train_box,
            model_type_radio,
            # llama config
            llama_ckpt,
            llama_base_config,
            llama_lr_slider,
            llama_maxsteps_slider,
            llama_data_num_workers_slider,
            llama_data_batch_size_slider,
            llama_data_max_length_slider,
            llama_precision_dropdown,
            llama_check_interval_slider,
            llama_grad_batches,
            llama_use_speaker,
            llama_use_lora,
        ],
        outputs=[train_error],
    )
    if_tb.change(
        fn=tensorboard_process,
        inputs=[if_tb, tb_dir, tb_host, tb_port],
        outputs=[train_error],
    )
    tb_dir.change(fn=fresh_tb_dir, inputs=[], outputs=[tb_dir])
    infer_decoder_model.change(
        fn=fresh_decoder_model, inputs=[], outputs=[infer_decoder_model]
    )
    infer_llama_model.change(
        fn=fresh_llama_model, inputs=[], outputs=[infer_llama_model]
    )
    llama_weight.change(fn=fresh_llama_model, inputs=[], outputs=[llama_weight])
    admit_btn.click(
        fn=check_files,
        inputs=[train_box, tree_slider, label_model, label_device],
        outputs=[error, file_markdown],
    )
    fresh_btn.click(
        fn=new_explorer, inputs=[train_box, tree_slider], outputs=[file_markdown]
    )
    llama_use_lora.change(
        fn=fresh_llama_ckpt, inputs=[llama_use_lora], outputs=[llama_ckpt]
    )
    llama_ckpt.change(
        fn=fresh_llama_ckpt, inputs=[llama_use_lora], outputs=[llama_ckpt]
    )
    lora_weight.change(
        fn=lambda: gr.Dropdown(choices=list_lora_llama_models()),
        inputs=[],
        outputs=[lora_weight],
    )
    llama_lora_merge_btn.click(
        fn=llama_lora_merge,
        inputs=[llama_weight, lora_llama_config, lora_weight, llama_lora_output],
        outputs=[train_error],
    )
    llama_quantify_btn.click(
        fn=llama_quantify,
        inputs=[llama_weight_to_quantify, quantify_mode],
        outputs=[train_error],
    )
    infer_checkbox.change(
        fn=change_infer,
        inputs=[
            infer_checkbox,
            infer_host_textbox,
            infer_port_textbox,
            infer_decoder_model,
            infer_decoder_config,
            infer_llama_model,
            infer_compile,
        ],
        outputs=[infer_error],
    )

demo.launch(inbrowser=True)
