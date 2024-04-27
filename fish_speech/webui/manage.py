from __future__ import annotations

import html
import json
import os
import platform
import random
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

from fish_speech.webui.launch_utils import Seafoam, versions_html

PYTHON = os.path.join(os.environ.get("PYTHON_FOLDERPATH", ""), "python")
sys.path.insert(0, "")
print(sys.path)
cur_work_dir = Path(os.getcwd()).resolve()
print("You are in ", str(cur_work_dir))
config_path = cur_work_dir / "fish_speech" / "configs"
vqgan_yml_path = config_path / "vqgan_finetune.yaml"
llama_yml_path = config_path / "text2semantic_sft.yaml"

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


def kill_process(pid):
    if system == "Windows":
        cmd = "taskkill /t /f /pid %s" % pid
        # os.system(cmd)
        subprocess.run(cmd)
    else:
        kill_proc_tree(pid)


def change_label(if_label):
    global p_label
    if if_label == True and p_label == None:
        cmd = ["asr-label-win-x64.exe"]
        yield f"打标工具WebUI已开启, 访问：http://localhost:{3000}"
        p_label = subprocess.Popen(cmd, shell=True, env=env)
    elif if_label == False and p_label != None:
        kill_process(p_label.pid)
        p_label = None
        yield "打标工具WebUI已关闭"


def change_infer(
    if_infer, host, port, infer_vqgan_model, infer_llama_model, infer_compile
):
    global p_infer
    if if_infer == True and p_infer == None:
        env = os.environ.copy()

        env["GRADIO_SERVER_NAME"] = host
        env["GRADIO_SERVER_PORT"] = port
        # 启动第二个进程
        yield build_html_ok_message(f"推理界面已开启, 访问 http://{host}:{port}")
        p_infer = subprocess.Popen(
            [
                PYTHON,
                "tools/webui.py",
                "--vqgan-checkpoint-path",
                infer_vqgan_model,
                "--llama-checkpoint-path",
                infer_llama_model,
                "--tokenizer",
                "checkpoints",
            ]
            + (["--compile"] if infer_compile == "Yes" else []),
            env=env,
        )

    elif if_infer == False and p_infer != None:
        kill_process(p_infer.pid)
        p_infer = None
        yield build_html_error_message("推理界面已关闭")


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


def add_item(folder: str, method: str, filelist: str, label_lang: str):
    folder = folder.strip(" ").strip('"')
    filelist = filelist.strip(" ").strip('"')

    folder_path = Path(folder)
    filelist_path = Path(filelist)

    if folder and folder not in items and data_pre_output not in folder_path.parents:
        if folder_path.is_dir():
            items.append(folder)
            dict_items[folder] = dict(
                type="folder", method=method, label_lang=label_lang
            )
        elif folder:
            err = folder
            return gr.Checkboxgroup(choices=items), build_html_error_message(
                f"添加文件夹路径无效: {err}"
            )

    if (
        filelist
        and filelist not in items
        and data_pre_output not in filelist_path.parents
    ):
        if filelist_path.is_file():
            items.append(filelist)
            dict_items[filelist] = dict(
                type="file", method=method, label_lang=label_lang
            )
        elif filelist:
            err = filelist
            return gr.Checkboxgroup(choices=items), build_html_error_message(
                f"添加文件路径无效: {err}"
            )

    formatted_data = json.dumps(dict_items, ensure_ascii=False, indent=4)
    logger.info(formatted_data)
    return gr.Checkboxgroup(choices=items), build_html_ok_message("添加文件(夹)路径成功!")


def remove_items(selected_items):
    global items, dict_items
    to_remove = [item for item in items if item in selected_items]
    for item in to_remove:
        del dict_items[item]
    items = [item for item in items if item in dict_items.keys()]
    formatted_data = json.dumps(dict_items, ensure_ascii=False, indent=4)
    logger.info(formatted_data)
    return gr.Checkboxgroup(choices=items, value=[]), build_html_ok_message(
        "删除文件(夹)路径成功!"
    )


def show_selected(options):
    selected_options = ", ".join(options)
    return f"你选中了: {selected_options}" if options else "你没有选中任何选项"


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
            if method == "复制一份":
                shutil.copy(original_wav_path, target_wav_path)
            else:
                shutil.move(original_wav_path, target_wav_path.parent)

            original_lab_path = original_wav_path.with_suffix(".lab")
            target_lab_path = (
                wav_root
                / original_wav_path.parent.name
                / original_wav_path.with_suffix(".lab").name
            )
            if target_lab_path.is_file():
                continue
            if method == "复制一份":
                shutil.copy(original_lab_path, target_lab_path)
            else:
                shutil.move(original_lab_path, target_lab_path.parent)

    if method == "直接移动":
        with list_file_path.open("w", encoding="utf-8") as file:
            file.writelines("\n".join(lst))

    del lst
    return build_html_ok_message("使用filelist")


def check_files(data_path: str, max_depth: int, label_model: str, label_device: str):
    dict_to_language = {"中文": "ZH", "英文": "EN", "日文": "JP", "不打标": "WTF"}

    global dict_items
    data_path = Path(data_path)
    for item, content in dict_items.items():
        item_path = Path(item)
        tar_path = data_path / item_path.name

        if content["type"] == "folder" and item_path.is_dir():
            cur_lang = dict_to_language[content["label_lang"]]
            if cur_lang != "WTF":
                try:
                    subprocess.run(
                        [
                            PYTHON,
                            "tools/whisper_asr.py",
                            "--model-size",
                            label_model,
                            "--device",
                            label_device,
                            "--audio-dir",
                            item_path,
                            "--save-dir",
                            item_path,
                            "--language",
                            cur_lang,
                        ],
                        env=env,
                    )
                except Exception:
                    print("Transcription error occured")

            if content["method"] == "复制一份":
                os.makedirs(tar_path, exist_ok=True)
                shutil.copytree(
                    src=str(item_path), dst=str(tar_path), dirs_exist_ok=True
                )
            elif not tar_path.is_dir():
                shutil.move(src=str(item_path), dst=str(tar_path))

        elif content["type"] == "file" and item_path.is_file():
            list_copy(item_path, content["method"])

    return build_html_ok_message("文件移动完毕"), new_explorer(data_path, max_depth=max_depth)


def train_process(
    data_path: str,
    option: str,
    # vq-gan config
    vqgan_lr,
    vqgan_maxsteps,
    vqgan_data_num_workers,
    vqgan_data_batch_size,
    vqgan_data_val_batch_size,
    vqgan_precision,
    vqgan_check_interval,
    # llama config
    llama_lr,
    llama_maxsteps,
    llama_limit_val_batches,
    llama_data_num_workers,
    llama_data_batch_size,
    llama_data_max_length,
    llama_precision,
    llama_check_interval,
):
    if option == "VQGAN" or option == "all":
        subprocess.run(
            [
                PYTHON,
                "tools/vqgan/create_train_split.py",
                str(data_pre_output.relative_to(cur_work_dir)),
            ]
        )
        train_cmd = [
            PYTHON,
            "fish_speech/train.py",
            "--config-name",
            "vqgan_finetune",
            f"model.optimizer.lr={vqgan_lr}",
            f"trainer.max_steps={vqgan_maxsteps}",
            f"data.num_workers={vqgan_data_num_workers}",
            f"data.batch_size={vqgan_data_batch_size}",
            f"data.val_batch_size={vqgan_data_val_batch_size}",
            f"trainer.precision={vqgan_precision}",
            f"trainer.val_check_interval={vqgan_check_interval}",
            f"train_dataset.filelist={str(data_pre_output / 'vq_train_filelist.txt')}",
            f"val_dataset.filelist={str(data_pre_output / 'vq_val_filelist.txt')}",
        ]
        logger.info(train_cmd)
        subprocess.run(train_cmd)

    if option == "LLAMA" or option == "all":
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
                "vqgan_pretrain",
                "--checkpoint-path",
                "checkpoints/vq-gan-group-fsq-2x1024.pth",
            ]
        )

        subprocess.run(
            [
                PYTHON,
                "tools/llama/build_dataset.py",
                "--input",
                str(data_pre_output),
                "--num-workers",
                "16",
            ]
        )

        protos_list = [
            str(file) for file in Path("data/quantized-dataset-ft").glob("*.protos")
        ]
        train_cmd = [
            PYTHON,
            "fish_speech/train.py",
            "--config-name",
            "text2semantic_sft",
            "model@model.model=dual_ar_2_codebook_medium",
            "tokenizer.pretrained_model_name_or_path=checkpoints",
            f"train_dataset.proto_files={str(protos_list)}",
            f"val_dataset.proto_files={str(protos_list)}",
            f"model.optimizer.lr={llama_lr}",
            f"trainer.max_steps={llama_maxsteps}",
            f"trainer.limit_val_batches={llama_limit_val_batches}",
            f"data.num_workers={llama_data_num_workers}",
            f"data.batch_size={llama_data_batch_size}",
            f"max_length={llama_data_max_length}",
            f"trainer.precision={llama_precision}",
            f"trainer.val_check_interval={llama_check_interval}",
        ]
        logger.info(train_cmd)
        subprocess.run(train_cmd)

    return build_html_ok_message("训练终止")


init_vqgan_yml = load_yaml_data_in_fact(vqgan_yml_path)
init_llama_yml = load_yaml_data_in_fact(llama_yml_path)

with gr.Blocks(
    head="<style>\n" + css + "\n</style>",
    js=js,
    theme=seafoam,
    analytics_enabled=False,
    title="Fish-Speech 鱼语",
) as demo:
    with gr.Row():
        with gr.Column():
            with gr.Tab("\U0001F4D6 数据集准备"):
                with gr.Row():
                    textbox = gr.Textbox(
                        label="\U0000270F 输入音频&转写源文件夹路径",
                        info="音频装在一个以说话人命名的文件夹内作为区分",
                        interactive=True,
                    )
                    transcript_path = gr.Textbox(
                        label="\U0001F4DD 转写文本filelist所在路径",
                        info="支持 Bert-Vits2 / GPT-SoVITS 格式",
                        interactive=True,
                    )
                with gr.Row(equal_height=False):
                    with gr.Column():
                        output_radio = gr.Radio(
                            label="\U0001F4C1 选择源文件(夹)处理方式",
                            choices=["复制一份", "直接移动"],
                            value="复制一份",
                            interactive=True,
                        )
                    with gr.Column():
                        error = gr.HTML(label="错误信息")
                        if_label = gr.Checkbox(
                            label="是否开启打标WebUI", scale=0, show_label=True
                        )
                with gr.Row():
                    add_button = gr.Button("\U000027A1提交到处理区", variant="primary")
                    remove_button = gr.Button("\U000026D4 取消所选内容")

                with gr.Row():
                    label_device = gr.Dropdown(
                        label="打标设备",
                        info="建议使用cuda, 实在是低配置再用cpu",
                        choices=["cpu", "cuda"],
                        value="cuda",
                        interactive=True,
                    )
                    label_model = gr.Dropdown(
                        label="打标模型大小",
                        info="显存10G以上用large, 5G用medium, 2G用small",
                        choices=["large", "medium", "small"],
                        value="small",
                        interactive=True,
                    )
                    label_radio = gr.Dropdown(
                        label="(可选)打标语言",
                        info="如果没有音频对应的文本，则进行辅助打标, 支持.txt或.lab格式",
                        choices=["中文", "日文", "英文", "不打标"],
                        value="不打标",
                        interactive=True,
                    )

            with gr.Tab("\U0001F6E0 训练配置项"):  # hammer
                with gr.Column():
                    with gr.Row():
                        model_type_radio = gr.Radio(
                            label="选择要训练的模型类型",
                            interactive=True,
                            choices=["VQGAN", "LLAMA", "all"],
                            value="all",
                        )
                    with gr.Row():
                        with gr.Accordion("VQGAN配置项", open=False):
                            with gr.Row(equal_height=False):
                                vqgan_lr_slider = gr.Slider(
                                    label="初始学习率",
                                    interactive=True,
                                    minimum=1e-5,
                                    maximum=1e-4,
                                    step=1e-5,
                                    value=init_vqgan_yml["model"]["optimizer"]["lr"],
                                )
                                vqgan_maxsteps_slider = gr.Slider(
                                    label="训练最大步数",
                                    interactive=True,
                                    minimum=1000,
                                    maximum=100000,
                                    step=1000,
                                    value=init_vqgan_yml["trainer"]["max_steps"],
                                )

                            with gr.Row(equal_height=False):
                                vqgan_data_num_workers_slider = gr.Slider(
                                    label="num_workers",
                                    interactive=True,
                                    minimum=1,
                                    maximum=16,
                                    step=1,
                                    value=init_vqgan_yml["data"]["num_workers"],
                                )

                                vqgan_data_batch_size_slider = gr.Slider(
                                    label="batch_size",
                                    interactive=True,
                                    minimum=1,
                                    maximum=32,
                                    step=1,
                                    value=init_vqgan_yml["data"]["batch_size"],
                                )
                            with gr.Row(equal_height=False):
                                vqgan_data_val_batch_size_slider = gr.Slider(
                                    label="val_batch_size",
                                    interactive=True,
                                    minimum=1,
                                    maximum=32,
                                    step=1,
                                    value=init_vqgan_yml["data"]["val_batch_size"],
                                )
                                vqgan_precision_dropdown = gr.Dropdown(
                                    label="训练精度",
                                    interactive=True,
                                    choices=["32", "bf16-true", "bf16-mixed"],
                                    value=str(init_vqgan_yml["trainer"]["precision"]),
                                )
                            with gr.Row(equal_height=False):
                                vqgan_check_interval_slider = gr.Slider(
                                    label="每n步保存一个模型",
                                    interactive=True,
                                    minimum=500,
                                    maximum=10000,
                                    step=500,
                                    value=init_vqgan_yml["trainer"][
                                        "val_check_interval"
                                    ],
                                )

                    with gr.Row():
                        with gr.Accordion("LLAMA配置项", open=False):
                            with gr.Row(equal_height=False):
                                llama_lr_slider = gr.Slider(
                                    label="初始学习率",
                                    interactive=True,
                                    minimum=1e-5,
                                    maximum=1e-4,
                                    step=1e-5,
                                    value=init_llama_yml["model"]["optimizer"]["lr"],
                                )
                                llama_maxsteps_slider = gr.Slider(
                                    label="训练最大步数",
                                    interactive=True,
                                    minimum=1000,
                                    maximum=100000,
                                    step=1000,
                                    value=init_llama_yml["trainer"]["max_steps"],
                                )
                            with gr.Row(equal_height=False):
                                llama_limit_val_batches_slider = gr.Slider(
                                    label="limit_val_batches",
                                    interactive=True,
                                    minimum=1,
                                    maximum=20,
                                    step=1,
                                    value=init_llama_yml["trainer"][
                                        "limit_val_batches"
                                    ],
                                )
                                llama_data_num_workers_slider = gr.Slider(
                                    label="num_workers",
                                    minimum=0,
                                    maximum=16,
                                    step=1,
                                    value=init_llama_yml["data"]["num_workers"]
                                    if sys.platform == "linux"
                                    else 0,
                                )
                            with gr.Row(equal_height=False):
                                llama_data_batch_size_slider = gr.Slider(
                                    label="batch_size",
                                    interactive=True,
                                    minimum=1,
                                    maximum=32,
                                    step=1,
                                    value=init_llama_yml["data"]["batch_size"],
                                )
                                llama_data_max_length_slider = gr.Slider(
                                    label="max_length",
                                    interactive=True,
                                    minimum=1024,
                                    maximum=4096,
                                    step=128,
                                    value=init_llama_yml["max_length"],
                                )
                            with gr.Row(equal_height=False):
                                llama_precision_dropdown = gr.Dropdown(
                                    label="训练精度",
                                    interactive=True,
                                    choices=["32", "bf16-true", "16-mixed"],
                                    value="bf16-true",
                                )
                                llama_check_interval_slider = gr.Slider(
                                    label="每n步保存一个模型",
                                    interactive=True,
                                    minimum=500,
                                    maximum=10000,
                                    step=500,
                                    value=init_llama_yml["trainer"][
                                        "val_check_interval"
                                    ],
                                )

            with gr.Tab("\U0001F9E0 进入推理界面"):
                with gr.Column():
                    with gr.Row():
                        with gr.Accordion(label="\U0001F5A5 推理服务器配置", open=False):
                            with gr.Row():
                                infer_host_textbox = gr.Textbox(
                                    label="Webui启动服务器地址", value="127.0.0.1"
                                )
                                infer_port_textbox = gr.Textbox(
                                    label="Webui启动服务器端口", value="7862"
                                )
                            with gr.Row():
                                infer_vqgan_model = gr.Textbox(
                                    label="VQGAN模型位置",
                                    placeholder="填写pth/ckpt文件路径",
                                    value="checkpoints/vq-gan-group-fsq-2x1024.pth",
                                )
                            with gr.Row():
                                infer_llama_model = gr.Textbox(
                                    label="LLAMA模型位置",
                                    placeholder="填写pth/ckpt文件路径",
                                    value="checkpoints/text2semantic-medium-v1-2k.pth",
                                )
                            with gr.Row():
                                infer_compile = gr.Radio(
                                    label="是否编译模型？", choices=["Yes", "No"], value="Yes"
                                )

                    with gr.Row():
                        infer_checkbox = gr.Checkbox(label="是否打开推理界面")
                        infer_error = gr.HTML(label="推理界面错误信息")

        with gr.Column():
            train_error = gr.HTML(label="训练时的报错信息")
            checkbox_group = gr.CheckboxGroup(
                label="\U0001F4CA 数据源列表",
                info="左侧输入文件夹所在路径或filelist。无论是否勾选，在此列表中都会被用以后续训练。",
                elem_classes=["data_src"],
            )
            train_box = gr.Textbox(
                label="数据预处理文件夹路径", value=str(data_pre_output), interactive=False
            )
            model_box = gr.Textbox(
                label="\U0001F4BE 模型输出路径",
                value=str(default_model_output),
                interactive=False,
            )

            with gr.Accordion(
                "查看预处理文件夹状态 (滑块为显示深度大小)",
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
                    "\U00002705 文件预处理", scale=0, min_width=160, variant="primary"
                )
                fresh_btn = gr.Button("\U0001F503", scale=0, min_width=80)
                help_button = gr.Button("\U00002753", scale=0, min_width=80)  # question
                train_btn = gr.Button("训练启动!", variant="primary")

    footer = load_data_in_raw("fish_speech/webui/html/footer.html")
    footer = footer.format(
        versions=versions_html(),
        api_docs="https://speech.fish.audio/inference/#http-api",
    )
    gr.HTML(footer, elem_id="footer")

    add_button.click(
        fn=add_item,
        inputs=[textbox, output_radio, transcript_path, label_radio],
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
    train_btn.click(
        fn=train_process,
        inputs=[
            train_box,
            model_type_radio,
            # vq-gan config
            vqgan_lr_slider,
            vqgan_maxsteps_slider,
            vqgan_data_num_workers_slider,
            vqgan_data_batch_size_slider,
            vqgan_data_val_batch_size_slider,
            vqgan_precision_dropdown,
            vqgan_check_interval_slider,
            # llama config
            llama_lr_slider,
            llama_maxsteps_slider,
            llama_limit_val_batches_slider,
            llama_data_num_workers_slider,
            llama_data_batch_size_slider,
            llama_data_max_length_slider,
            llama_precision_dropdown,
            llama_check_interval_slider,
        ],
        outputs=[train_error],
    )
    admit_btn.click(
        fn=check_files,
        inputs=[train_box, tree_slider, label_model, label_device],
        outputs=[error, file_markdown],
    )
    fresh_btn.click(
        fn=new_explorer, inputs=[train_box, tree_slider], outputs=[file_markdown]
    )

    infer_checkbox.change(
        fn=change_infer,
        inputs=[
            infer_checkbox,
            infer_host_textbox,
            infer_port_textbox,
            infer_vqgan_model,
            infer_llama_model,
            infer_compile,
        ],
        outputs=[infer_error],
    )

demo.launch(inbrowser=True)