export WANDB_API_KEY=8b2885096bc8e2e0291c1e6aec2de6f864bba024

huggingface-cli login  --token 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

git clone https://github.com/YYT-t/xyzo.git
cd xyzo

conda env create -f environment.yaml
conda activate yy

python3 -m pip install lm_eval

export MASTER_ADDR=$ARNOLD_WORKER_0_HOST
export MASTER_PORT=$(echo "$ARNOLD_WORKER_0_PORT" | cut -d "," -f 1)
echo "MASTER_ADDR: $MASTER_ADDR, MASTER_PORT: $MASTER_PORT"

ACCELERATE_LOG_LEVEL=info accelerate launch --main_process_ip $MASTER_ADDR —main_process_port $MASTER_PORT --num_machines 2 --machine_rank $ARNOLD_ID E_step_ent.py \
    --model_name google/gemma-2-9b-it   \
    --task_type math_gsm \
    --deepspeed ./deepspeed_configs/deepspeed_3.json \
    --output_suffix "" \
    --ent_coeff 0.02 \
    --do_sample False \
    --num_beams 1\
    --temperature 0.8 \
    --num_train_epochs 1 \
    --max_length 256 \
    --save_strategy steps \
    --save_every_steps 50 \
    --gradient_accumulation_steps 2 \
    --per_device_train_batch_size 4 \
    --wandb_project E_step_ent \



# export WANDB_API_KEY=8b2885096bc8e2e0291c1e6aec2de6f864bba024

# huggingface-cli login  --token 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

# git clone https://github.com/YYT-t/xyzo.git
# cd xyzo

# conda env create -f environment.yaml
# conda activate yy

# export MASTER_ADDR=$ARNOLD_WORKER_0_HOST
# export MASTER_PORT=$(echo "$ARNOLD_WORKER_0_PORT" | cut -d "," -f 1)
# echo "MASTER_ADDR: $MASTER_ADDR, MASTER_PORT: $MASTER_PORT"


# ACCELERATE_LOG_LEVEL=info accelerate launch --main_process_ip $MASTER_ADDR —main_process_port $MASTER_PORT --num_machines 2 --machine_rank $ARNOLD_ID E_step_ent.py \
#     --model_name google/gemma-2-9b-it   \
#     --task_type math_gsm \
#     --deepspeed ./deepspeed_configs/deepspeed_3.json \
#     --output_suffix "" \
#     --ent_coeff 0.01 \
#     --do_sample False \
#     --num_beams 1\
#     --temperature 0.8 \
#     --num_train_epochs 1 \
#     --max_length 256 \
#     --save_strategy steps \
#     --save_every_steps 50 \
#     --gradient_accumulation_steps 2 \
#     --per_device_train_batch_size 4 \
#     --wandb_project E_step_ent \