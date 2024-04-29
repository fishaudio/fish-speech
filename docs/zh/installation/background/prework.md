首先，请你参照[conda教程](zh/installation/background/conda.md)和[git教程](zh/installation/background/git.md)安装好miniconda和git并**确保配置了环境变量**。

!!! warning "警告"
    我们强烈不建议你把教程中的程序安装到C盘

## 克隆项目仓库
在你想要安装fish-speech的地方右键打开终端，，输入
```
git clone https://github.com/fishaudio/fish-speech.git
```
如果你所在的网络环境无法链接github，你可以可以前往[fish-speech的github库]手动下载.zip文件并解压到你指定的文件目录下。

<p align="center">
​<img src="/assets/images/git_zip.png" alt="git_zip" style="zoom:50%;" />
</p>

## 安装conda环境
继续在终端中进行操作
```
conda create -n fish-speech python=3.11
```
运行成功后如下所示:
<p align="center">
​<img src="/assets/images/conda_success.png" alt="conda_success" style="zoom:65%;" />
</p>

继续在终端中输入
```
conda activate fish-speech
```
至此，你已经成功激活了conda环境

## 下载预训练权重

**至此，你已经完成了全部的前置工作**
