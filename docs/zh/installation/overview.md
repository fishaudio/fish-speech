## 配置要求
- GPU内存: 2GB (用于推理), 16GB (用于微调)
- 系统: Linux (全部功能), Windows (仅推理, 不支持 `flash-attn`, 不支持 `torch.compile`)

你的GPU需要支持cuda，也就是说你需要一张nvidia的gpu

因此, 我们强烈建议 Windows 用户使用 [WSL2](zh/installation/wsl.md) 或 [docker](zh/installation/docker.md) 来运行代码库.
!!! tips
    考虑到很多同学可能没有足够的配置，我们也准备了使用AutoDL云服务器进行安装和微调的教程。

!!! warning
    如果你对计算机或者AI部署相关知识不熟悉，请务必确保你看过[**背景知识**](zh/installation/background/overview.md)再开始后面的操作。
