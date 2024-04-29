git是一个版本管理工具，我们使用git来拉取github上我们的项目。

首先你需要下载git

[windows版官网下载链接](https://gitforwindows.org/)

linux执行命令`sudo apt install git`

!!! warning "警告"
    在安装过程中请务必选择添加到环境变量或者是添加到PATH

安装完成后，在终端中输入`git -v`，如果你能看到版本信息，说明安装成功了

之后你需要配置一下`git config --global`，在git bash或者终端中执行以下命令
```
git config --global user.name "fish audio"
git config --global user.email user@fishaudio.com
```

恭喜你完成了git的配置