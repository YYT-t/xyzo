huggingface-cli login --token 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


ACCELERATE_LOG_LEVEL=info CUDA_VISIBLE_DEVICES=1,2,3,4  accelerate launch --main_process_port 29052 z_sa_4.py \
    --model_name google/gemma-2-2b-it  \
    --train_set_path openai/gsm8k \
    --deepspeed ./deepspeed_configs/deepspeed_3.json\
    --output_path ./Q_models/tt_4_gemma\
    --max_length 256 \
    --save_every_steps 50 \
    --per_device_train_batch_size 4 \

# CUDA_VISIBLE_DEVICES=4  python z.py --model_name google/gemma-2-2b-it --train_set_path openai/gsm8k #--deepspeed ./deepspeed_configs/deepspeed_3.jsona

# #!/bin/bash

# # 指定使用的 GPU 数量
# NUM_GPUS=4  # 根据您的硬件情况调整

# # DeepSpeed 配置文件的路径
# DEEPSPEED_CONFIG="deepspeed_configs/deepspeed_3.json"

# # 运行训练脚本
# deepspeed --num_gpus=$NUM_GPUS z_sa.py \
#     --deepspeed $DEEPSPEED_CONFIG \
#     --model_name "google/gemma-2-2b-it" \
#     --train_set_path "openai/gsm8k" \
#     --output_path "./Q_models/debug" \
#     --per_device_train_batch_size 4 \
#     --gradient_accumulation_steps 4 \
#     --learning_rate 5e-7 \
#     --weight_decay 0.001 \
#     --num_train_epochs 1 \
#     --max_length 10 \
#     --gradient_checkpointing True \
#     --bf16 True \
#     --optim "adamw_torch" \
#     --lr_scheduler_type "cosine" \
#     --save_every_steps 999999 \
#     --eval_every_steps 999999
