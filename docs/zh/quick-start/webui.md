在运行 WebUI 之前, 你需要先启动 HTTP 服务, 如上所述.

随后你可以使用以下命令来启动 WebUI:

```bash
python fish_speech/webui/app.py
```

或附带参数来启动 WebUI:

```bash
# 以临时环境变量的方式启动:
GRADIO_SERVER_NAME=127.0.0.1 GRADIO_SERVER_PORT=7860 python fish_speech/webui/app.py
```

祝大家玩得开心!
