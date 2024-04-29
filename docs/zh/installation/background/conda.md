conda环境是一个和其他环境隔离的虚拟python环境，旨在提供更好的包管理和版本管理。

既然你已经点开了这个页面，说明你应该是完全没有接触过conda环境的小白，我们推荐使用miniconda，其能提供更简易更易用的conda环境

[miniconda官方下载入口](https://docs.anaconda.com/free/miniconda/)

[windows版官网下载链接](https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe)

[linux版官网下载链接](https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh)

考虑到你可能只能使用国内网络环境，下面是anaconda的国内镜像站地址：

[anaconda国内镜像站链接](https://mirrors.tuna.tsinghua.edu.cn/anaconda/archive/)

!!! warning "警告"
    在安装过程中请务必选择添加到环境变量或者是添加到PATH

下面给出一些最基本的conda指令:
```
# 创建一个conda环境并指定python版本
conda create -n 你的环境名 python=3.XX

# 激活存在的conda环境
conda activate 你的环境名

# 退出已激活的conda环境到base环境
conda deactivate

# 删除conda环境
# 删除前请先退出该环境
conda remove --name 你的环境名 --all
```
!!! note "提示"
    在遇到需要输入的时候一路回车就可以了