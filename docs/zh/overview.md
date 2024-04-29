## fish-speech的功能
fish-speech是一个经过预训练的语音生成模型，主要用于生成具有二次元风格的语音。其只需要几十秒的语音数据就能生成较为真实的角色语音。我们提供了一个1B参数的预训练模型，你可以通过[本地部署模型](zh/installation/overview.md)来让自己喜欢角色生成想要的话。

我们同时支持[终端命令行](zh/quick-start/terminal.md), [http api接口](zh/quick-start/api.md), 以及 [webui](zh/quick-start/webui.md) 三种方式进行推理工作。旅行者、开拓者、舰长、博士、sensei、猎魔人、喵喵露、V可以选择自己喜欢的方式进行推理工作。

同时考虑到使用本项目进行Text to Speech任务的旅行者、开拓者、舰长、博士、sensei、猎魔人、喵喵露、V可能并不熟悉计算机和AI模型相关内容，我们也准备了详细的[背景知识](zh/installation/background/overview.md)和[多平台的安装指南](zh/installation/linux.md)，希望旅行者、开拓者、舰长、博士、sensei、猎魔人、喵喵露、V可以生成自己喜欢的语音。

我们建议在linux环境或者容器（如docker和wsl）内安装部署我们项目并进行推理和微调，如果你一定要使用windows环境，请务必使用我们的整合包并按照[windows安装](zh/installation/windows.md)文档进行操作。

## fish-speech和Bert-vits2相比有哪些区别和优势
fish-speech是一个全新的自回归TTS模型，相比于Bert-vits2而言，fish-speech能做到更好的语音生成，更多的语言支持，更长的单个语音生成以及更快的生成速度。

## 项目架构
本项目使用类似VALL-E的语音生成架构，使用微调特化的Llama作为大模型基座，生成量化的梅尔token。之后通过VQ-GAN架构的声码器转换为真实的音频，具体的项目架构请参考[架构](zh/frameworks/overview.md)部分。

## 模型微调
如果你对我们的预训练模型效果不满意，我们提供了基于Lora和SFT的微调功能，你可以加上自己的数据集对我们的预训练模型进行微调。

**祝大家玩得开心**