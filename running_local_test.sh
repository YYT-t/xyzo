export WANDB_API_KEY=8b2885096bc8e2e0291c1e6aec2de6f864bba024

CUDA_VISIBLE_DEVICES=1,2,3,4 ACCELERATE_LOG_LEVEL=info accelerate launch   E_step_ent_test.py \
    --model_name google/gemma-2-2b-it  \
    --task_type math_gsm \
    --deepspeed ./deepspeed_configs/deepspeed_3.json \
    --output_suffix "" \
    --ent_coeff 0.05 \
    --num_beams 1\
    --do_sample False \
    --temperature 1.0 \
    --num_train_epochs 1 \
    --max_length 256 \
    --save_strategy steps \
    --save_every_steps 50 \
    --per_device_train_batch_size 4
