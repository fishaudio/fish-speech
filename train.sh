docker run --rm -it --gpus all \
    --ipc=host --shm-size=16g --ulimit memlock=-1 --ulimit stack=67108864 \
    -v $(pwd):/exp speech-llm-train
