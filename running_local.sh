export WANDB_API_KEY=8b2885096bc8e2e0291c1e6aec2de6f864bba024
huggingface-cli login  --token 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

CUDA_VISIBLE_DEVICES=0,1,2,3 ACCELERATE_LOG_LEVEL=info accelerate launch --main_process_port 29501 E_step_ent_tensor_rewnorm.py \
    --model_name google/gemma-2-9b-it  \
    --task_type math_gsm \
    --deepspeed ./deepspeed_configs/deepspeed_3.json \
    --output_suffix "" \
    --ent_coeff 0.00 \
    --num_beams 1\
    --do_sample False \
    --temperature 1.0 \
    --label_smoothing 0.001 \
    --use_template True \
    --num_train_epochs 1 \
    --max_length 256 \
    --save_strategy steps \
    --save_every_steps 30 \
    --gradient_accumulation_steps 4 \
    --per_device_train_batch_size 4 \
