#!/bin/bash
# cd em
# conda env create -f environment.yml
# conda activate sft

iter_num=3
TASK_ID=code_opencoder_edu
MODEL_FULL_NAME=deepseek-ai/deepseek-coder-6.7b-instruct
MODEL_NICKNAME=deepseek-coder-6.7b-instruct
DATASET_NAME=OpenCoder-LLM/opc-sft-stage2
path="./${MODEL_NICKNAME}"
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

    CUDA_VISIBLE_DEVICES=0,1,2,3 ACCELERATE_LOG_LEVEL=info accelerate launch --main_process_port $PORT1 E_step_ent.py \
    --model_name ${MODEL_FULL_NAME}  \
    --train_set_path ${DATASET_NAME} \
    --deepspeed ./deepspeed_configs/deepspeed_3.json \
    --output_suffix "" \
    --ent_coeff 0.05 \
    --num_beams 1 \
    --do_sample False \
    --temperature 0.8 \
    --num_train_epochs 1 \
    --max_length 256 \
    --save_every_steps 50 \
    --gradient_accumulation_steps 16 \
    --per_device_train_batch_size 1 \
    --model_path $e_model_dir \

    ## The full batch size is 512
    python inference.py --model_path "${e_model_dir}/final_ckpt" --dataset_path $dataset_path --iter i || exit 1
    accelerate launch --num_processes 4 m_sft.py --deepspeed deepspeed_configs/deepspeed_2.json --model_name $e_model_dir \
    --per_device_train_batch_size 1 --gradient_accumulation_steps 128 --train_set_path $dataset_path --output_dir $m_model_dir \
    --num_train_epochs 1 --Task_Type ${TASK_ID} --learning_rate 5e-5 \
    --hub_model_id $m_hub_id || exit 1
done