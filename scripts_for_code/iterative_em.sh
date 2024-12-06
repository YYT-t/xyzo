#!/bin/bash
# cd em
# conda env create -f environment.yml
# conda activate sft
export WANDB_API_KEY=84f03efa3815c8727157b1951519ce4b0f2a190a
wandb login --relogin 84f03efa3815c8727157b1951519ce4b0f2a190a
iter_num=3
TASK_ID=code_opencoder_edu
MODEL_FULL_NAME=deepseek-ai/deepseek-coder-6.7b-instruct
MODEL_NICKNAME=deepseek-coder-6.7b-instruct-v0
DATASET_NAME=OpenCoder-LLM/opc-sft-stage2
path="./${MODEL_NICKNAME}"
conda activate sft || conda env create -f environment_sft.yml
conda activate yy || conda env create -f environment.yaml
export HF_TOKEN=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
for i in $(seq 1 $iter_num); do
    mkdir $path
    e_input_model="${path}/m-model-iter-$((i-1))"
    e_model_dir="${path}/e-model-iter-$i"
    m_model_dir="${path}/m-model-iter-$i"
    m_hub_id="${MODEL_NICKNAME}-m-model-iter-$i"
    dataset_path="ZhangShenao/iterative-${TASK_ID}-iter$i"
    if [ "$i" -eq 1 ]; then
        e_input_model=${MODEL_FULL_NAME}
    else
        echo "iteration $i"
    fi
    conda activate yy
    ACCELERATE_LOG_LEVEL=info accelerate launch --main_process_port $PORT1 E_step_ent.py \
    --model_name ${MODEL_FULL_NAME}  \
    --train_set_path ${DATASET_NAME} \
    --deepspeed ./deepspeed_configs/deepspeed_3.json \
    --output_suffix "" \
    --ent_coeff 0.05 \
    --num_beams 1 \
    --do_sample False \
    --temperature 0.8 \
    --num_train_epochs 3 \
    --max_length 256 \
    --save_every_steps 50 \
    --gradient_accumulation_steps 1 \
    --per_device_train_batch_size 16 \
    --model_path $e_model_dir \
    
    conda activate sft
    ## The full batch size is 512
    python inference.py --model_path "${e_model_dir}/final_ckpt" --dataset_path $dataset_path --iter i || exit 1
    accelerate launch --num_processes 8 m_sft.py --deepspeed deepspeed_configs/deepspeed_2.json --model_name $e_model_dir \
    --per_device_train_batch_size 8 --gradient_accumulation_steps 8 --train_set_path $dataset_path --output_dir $m_model_dir \
    --num_train_epochs 3 --Task_Type ${TASK_ID} --learning_rate 5e-5 \
    --hub_model_id $m_hub_id || exit 1
done