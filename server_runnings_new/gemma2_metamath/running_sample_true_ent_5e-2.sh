export WANDB_API_KEY=8b2885096bc8e2e0291c1e6aec2de6f864bba024

huggingface-cli login  --token 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

git clone https://github.com/YYT-t/xyzo.git
cd xyzo

conda env create -f environment.yaml
conda activate yy

ACCELERATE_LOG_LEVEL=info accelerate launch --main_process_port $PORT1 E_step_ent.py \
    --model_name google/gemma-2-9b-it   \
    --task_type math_metamath \
    --deepspeed ./deepspeed_configs/deepspeed_3.json \
    --output_suffix "" \
    --ent_coeff 0.05 \
    --do_sample True \
    --num_beams 1\
    --temperature 0.8 \
    --num_train_epochs 1 \
    --max_length 256 \
    --save_strategy steps \
    --save_every_steps 50 \
    --gradient_accumulation_steps 1 \
    --per_device_train_batch_size 16 \
    --wandb_project E_step_ent \