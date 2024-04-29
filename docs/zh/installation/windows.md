!!! warning "警告"
    我们非常不建议使用windows环境进行部署，如果你无法使用其他环境或者觉得wsl和docker上手很困难，请务必认真阅读下面的文档并按照步骤操作

??? info "测试环境"

!!! info "提示"
    我们强烈建议你先阅读[背景知识](zh/installation/background/overview.md)再开始后面的工作！

    我们强烈建议你先阅读[背景知识](zh/installation/background/overview.md)再开始后面的工作！

    我们强烈建议你先阅读[背景知识](zh/installation/background/overview.md)再开始后面的工作！

## 0.环境准备<a id="section0"></a>

- [LLVM](https://github.com/llvm/llvm-project/releases/download/llvmorg-17.0.4/LLVM-17.0.4-win64.exe) (需要编译模型，加速推理时必须下载安装)

  

下载好整合包以后，解压，[如何下载解压](https://www.bilibili.com/video/BV18z42167dQ/)

点击目录下的`start.bat`启动

<p align="center">
​	<img src="/assets/images/windows_installation/image-20240428203043710.png" alt="image-20240428203043710" />
</p>

弹出来的黑色窗口是`后端`，使用过程中不能关闭。点开后，程序应该自动会使用您机器上的默认浏览器打开一个`Fish-speech页面`。整个过程不超过5秒。

<p align="center">
​	<img src="/assets/images/windows_installation/image-20240428203145465.png" alt="image-20240428203145465" />
</p>

若迟迟未有页面打开，请您手动复制绿色框的地址到浏览器的导航栏，然后按下回车键。此时会打开一个新页面。

<p align="center">
​	<img src="/assets/images/windows_installation/image-20240428203504874.png" alt="image-20240428203504874" />
</p>

​	进入网页后：

<p align="center">
​	<img src="/assets/images/windows_installation/image-20240428213152978.png" alt="image-20240428213152978" />
</p>

简单说一下各部分区域构成，如下图所示，方便按图索骥：

<p align="center">
​	<img src="/assets/images/windows_installation/image-20240428213046888.png" alt="image-20240428213046888" />
</p>

1. banner（横幅）：进入网页后从左到右逐渐显示"Welcome to Fish-Speech"字样。以后可能变动。
2. 功能区: 在这里，你将决定数据集文件的来源，文本标签的修改，训练参数的调整、推理页面的设置。
3. 文件信息展示区：一般不可更改。指引你如何找到自己的预处理后的数据文件、训练后的模型文件所在路径。
4. 版本/作者信息。可以多多支持一下作者。
5. 欢迎更好的动效~

## 1. 准备数据集 <a id="section1"></a>

​	需要准备若干个文件夹，每个文件夹内存放如下类型的数据：

```cpp
.
├── SPK1
│   ├── 21.15-26.44.mp3
│   ├── 27.51-29.98.mp3
│   └── 30.1-32.71.mp3
└── SPK2
    └── 38.79-40.85.mp3
```

​	你需要将数据集转为以上格式, 并放到 `data` 下, 音频后缀可以为 `.mp3`, `.wav` 或 `.flac`.

​	如果一个文件夹内有音频文件的同时，还有同名的`.lab`或`.txt`文件，我们称之为`“完整”`数据集。若只有音频文件，则称之为`“不完整”`数据集。显然，完整的数据集可以直接参与训练。

​	下面看一个例子：
<p align="center">
​	<img src="/assets/images/windows_installation/image-20240428223627202.png" alt="image-20240428223627202" style="zoom:50%;" />
</p>

​	如上图所示，`胡桃`文件夹内有若干音频`.wav`文件，还有对应的`.lab`文件，也就是音频的”标签“。那么这就是一个”完整“数据文件夹。

​	标签里是一些文本，是对应音频的听写文件。音频说了什么，里面写什么就是了。像这样：

<p align="center">
​	<img src="/assets/images/windows_installation/image-20240428224021566.png" alt="image-20240428224021566" />
</p>
​	

​	如果你的数据集不完整，没有标签，可以考虑整合包内的`打标`（听写）功能：
<p align="center"> 
​	<img src="/assets/images/windows_installation/image-20240428224341386.png" alt="image-20240428224341386" style="zoom:50%;" />
</p>

如图，“功能区”中的红框部分是打标选项。确定目标文件夹说话人的语言，如果是`中文`，推荐如下配置，性能较好，配置要求不过高：

<p align="center">
​	<img src="/assets/images/windows_installation/image-20240428225907916.png" alt="image-20240428225907916" />
</p>

然后是复制文件夹所在路径，点击下图中的红色部分

<p align="center">
<img src="/assets/images/windows_installation/image-20240428230155396.png" alt="image-20240428230155396" style="zoom:50%;" />
</p>

变成下图的选中状态，按`Ctrl + C` 复制即可：

<p align="center">
​	<img src="/assets/images/windows_installation/image-20240428230436481.png" alt="image-20240428230436481" />
</p>


将复制的路径粘贴到“功能区”左上角的这个框里：

<p align="center">
​	<img src="/assets/images/windows_installation/image-20240428230640193.png" alt="image-20240428230640193" style="zoom:50%;" />
</p>

​	下面有文件夹的处理方式，默认是“复制一份”源数据集到“数据集预处理文件夹”中。

​	选择好后，点击中间的蓝底白字按钮`“提交到处理区”`。此时，这份数据集会被标记为“不完整”，需要听写打标。你可以持续重复上述过程许多次，直到把所有的“不完整”数据添加完成。

![image-20240428231003308](/assets/images/windows_installation/image-20240428231003308.png)

​	添加完成后，屏幕右侧的展示区中，“数据源列表”会更新。一个粉色的确认框代表一个要“处理“的数据集。如果发现一些数据集被错误添加了，可以点击确认框的白色区域，按下勾选。再点击左侧的取消所选内容。


<p align="center">
<img src="/assets/images/windows_installation/image-20240428231526990.png" alt="image-20240428231526990" style="zoom: 67%;" />
</p>

​	去除成功，如下图所示：

<p align="center">
<img src="/assets/images/windows_installation/image-20240428231621389.png" alt="image-20240428231621389" style="zoom:50%;" />
</p>

​	

​	如果你的数据是“完整”的，那么你可以直接复制文件夹的路径到`“输入音频&转写源文件所在路径”`里，再点击`提交到处理区`按钮即可。



​	当展示区的数据源列表中出现了所有你要处理的数据时，点击展示区下面的`“文件预处理”`按钮，程序会按需求进行打标整理，并放置在数据预处理文件夹里。到此，数据预处理的工作就完成了。

​	

## 2. 微调模型<a id="section2"></a>

​	对预训练模型不满意？考虑微调。通过对少量样本的学习，让模型生成更符合目标特征的声音。

​	完成第1节.`准备数据集`后，点击“功能区”上方的`“训练配置项”`。

<p align="center">
<img src="/assets/images/windows_installation/image-20240428233720728.png" alt="image-20240428233720728" style="zoom:50%;" />
</p>

​	如果你的配置足够好，显存内存很大(A100级别)，那么程序默认选择的`all`是最佳的。你可以直接点击右侧展示区，右下角的`“训练启动”`稍作等待即可。训练完成后，展示区上方会显示一行绿色小字`训练终止`。

<p align="center">
​	<img src="/assets/images/windows_installation/image-20240428234109105.png" alt="image-20240428234109105" style="zoom:50%;" />
</p>

​	一般情况下，都需要修改默认配置。各个参数的含义可以自行搜索，这里展示推荐配置。



### VQGAN

<p align="center">
​    	<img src="/assets/images/windows_installation/image-20240428234607478.png" alt="image-20240428234607478" style="zoom:50%;" />
</p>

### LLAMA

<p align="center">
​	<img src="/assets/images/windows_installation/image-20240428234837051.png" alt="image-20240428234837051" style="zoom:50%;" />

</p>
​	

## 3. Web界面推理<a id="section3"></a>

​	

![image-20240428235300141](/assets/images/windows_installation/image-20240428235300141.png)

​	默认用官方初始模型。如果用自己训练的模型，请复制results目录下的`.ckpt`文件`路径`到对应的框中。推荐30系及以上N卡开启`编译模型`选项。然后点击`是否打开推理界面`勾选框，开始推理吧。 