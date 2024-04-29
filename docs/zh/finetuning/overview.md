显然, 当你打开这个页面的时候, 你已经对预训练模型 few-shot 的效果不算满意. 你想要微调一个模型, 使得它在你的数据集上表现更好.  

`Fish Speech` 由两个模块组成: `VQGAN` 和 `LLAMA`. 

!!! info 
    你应该先进行如下测试来判断你是否需要微调 `VQGAN`:
    ```bash
    python tools/vqgan/inference.py -i test.wav
    ```
    该测试会生成一个 `fake.wav` 文件, 如果该文件的音色和说话人的音色不同, 或者质量不高, 你需要微调 `VQGAN`.

    相应的, 你可以参考 [推理](inference.md) 来运行 `generate.py`, 判断韵律是否满意, 如果不满意, 则需要微调 `LLAMA`.