CUDA_VISIBLE_DEVICES=1,2,3,4 ACCELERATE_LOG_LEVEL=info accelerate launch   E_step_ent.py \
    --model_name google/gemma-1.1-7b-it  \
    --task_type math_gsm \
    --deepspeed ./deepspeed_configs/deepspeed_3.json \
    --output_suffix "" \
    --ent_coeff 0.05 \
    --num_beams 1\
    --do_sample False \
    --temperature 1.0 \
    --num_train_epochs 3 \
    --max_length 256 \
    --save_every_steps 50 \
    --per_device_train_batch_size 4
