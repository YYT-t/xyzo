#!/bin/bash
# cd em
# conda env create -f environment.yml
# conda activate sft

iter_num=3
path="./gemma-2-9b"
export HF_TOKEN=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
for i in $(seq 1 $iter_num); do
    mkdir $path
    e_input_model="${path}/m-model-iter-$((i-1))"
    e_model_dir="${path}/e-model-iter-$i"
    m_model_dir="${path}/m-model-iter-$i"
    m_hub_id="gemma2-9b-it-m-model-iter-$i"
    dataset_path="ZhangShenao/iterative-metamath-iter$i"
    if [ "$i" -eq 1 ]; then
        e_input_model="google/gemma-2-9b-it"
    else
        echo "iteration $i"
    fi

    CUDA_VISIBLE_DEVICES=0,1,2,3  ACCELERATE_LOG_LEVEL=info accelerate launch E_step_ent.py \
    --model_name google/gemma-2-9b-it  \
    --task_type math_metamath \
    --deepspeed ./deepspeed_configs/deepspeed_3.json \
    --output_suffix "" \
    --ent_coeff 0.05 \
    --num_beams 1\
    --do_sample False \
    --temperature 0.8 \
    --num_train_epochs 1 \
    --max_length 256 \
    --save_every_steps 50 \
    --gradient_accumulation_steps 1 \
    --per_device_train_batch_size 16 \
    --model_path $e_model_dir \

#    python E_step_ent_metamath.py --TODO # input model is e_input_model, output is e_model_dir  || exit 1
    python inference.py --model_path "${e_model_dir}/final_ckpt" --dataset_path $dataset_path --iter $i || exit 1
    accelerate launch --num_processes 8 m_sft.py --deepspeed deepspeed_configs/deepspeed_2.json --model_name $e_model_dir --attn_implementation eager --per_device_train_batch_size 2 --gradient_accumulation_steps 8 --train_set_path $dataset_path --output_dir $m_model_dir --hub_model_id $m_hub_id || exit 1
done