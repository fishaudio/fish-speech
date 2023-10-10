export NCCL_P2P_DISABLE=1

hostfile=""
deepspeed --hostfile=$hostfile tools/tts/fine-tune.py \
    --deepspeed tools/tts/ds_config.json \
    --report_to "tensorboard" \
    --data_path "fishaudio/libritts-r-encodec" \
    --model_name_or_path "checkpoints/llama2-tiny-init" \
    --output_dir "results" \
    --model_max_length 4096 \
    --max_steps 500000 \
    --per_device_train_batch_size 32 \
    --gradient_accumulation_steps 1 \
    --save_strategy steps \
    --save_steps 10000 \
    --evaluation_strategy steps \
    --eval_steps 10000 \
    --learning_rate 1e-3 \
    --lr_scheduler_type cosine \
    --adam_beta1 0.9 \
    --adam_beta2 0.98 \
    --adam_epsilon 1e-8 \
    --max_grad_norm 1.0 \
    --weight_decay 1e-4 \
    --warmup_steps 10000 \
    --logging_steps 1 \
    --gradient_checkpointing True \
    --remove_unused_columns False \
    --use_lora False \
    --bf16 True \
    --tf32 True
