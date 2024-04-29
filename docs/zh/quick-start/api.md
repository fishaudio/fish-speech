运行以下命令来启动 HTTP 服务:

```bash
python -m zibai tools.api_server:app --listen 127.0.0.1:8000
# 推荐中国大陆用户运行以下命令来启动 HTTP 服务:
HF_ENDPOINT=https://hf-mirror.com python -m zibai tools.api_server:app --listen 127.0.0.1:8000
```

随后, 你可以在 `http://127.0.0.1:8000/docs` 中查看并测试 API.  
一般来说, 你需要先调用 `PUT /v1/models/default` 来加载模型, 然后调用 `POST /v1/models/default/invoke` 来进行推理.
具体的参数请参考 API 文档.