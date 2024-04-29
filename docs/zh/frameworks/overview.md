fish-speech的整体框架结构如下图所示
<p align="center">
<img src="/assets/images/diagram.png" width="75%">
</p>
fish-speech主要使用了两个模块，一个基于llama的mel token生成模块和一个基于VQ-gan的声码器模块。具体的训练方式是通过类似VALL-E的架构通过自回归的语言模型