#!/bin/bash
# cd em
# conda env create -f environment.yml
# conda activate sft

export CUDA_VISIBLE_DEVICES=0,1,2,3

conda env create -f environment.yaml
conda env create -f environment_sft.yaml


export HF_TOKEN=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
iter_num=3

company="google"
model_name="gemma-2-2b-it"

task_pre="math"
task_suf="gsm"

path="./${model_name}"
export HF_TOKEN=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
for i in $(seq 1 $iter_num); do
    conda activate yy
    
    mkdir $path
    e_input_model="${path}/m-model-iter-$((i-1))"
    e_model_dir="${path}/e-model-iter-$i"
    m_model_dir="${path}/m-model-iter-$i"
    m_hub_id="${model_name}-m-model-iter-$i"
    dataset_path="YYT-t/iterative-${task_suf}-iter$i"
    if [ "$i" -eq 1 ]; then
        e_input_model="${company}/${model_name}"
    else
        echo "iteration $i"
    fi

    # ACCELERATE_LOG_LEVEL=info accelerate launch E_step_ent_test.py \
    # --model_name $e_input_model \
    # --task_type "${task_pre}_${task_suf}" \
    # --deepspeed ./deepspeed_configs/deepspeed_3.json \
    # --output_suffix "" \
    # --ent_coeff 0.05 \
    # --num_beams 1\
    # --do_sample False \
    # --temperature 0.8 \
    # --num_train_epochs 1 \
    # --max_length 256 \
    # --save_every_steps 50 \
    # --gradient_accumulation_steps 1 \
    # --per_device_train_batch_size 16 \
    # --model_path $e_model_dir \
    # # --upload_to_hub False \

    python inference_test.py --model_path "${e_model_dir}/final_checkpoint"  --task_type "${task_pre}_${task_suf}" --dataset_path $dataset_path --iter $i
    conda deactivate

    conda activate sft_debug
    accelerate launch m_sft_test.py --deepspeed deepspeed_configs/deepspeed_2.json --model_name $e_input_model --attn_implementation eager --per_device_train_batch_size 4 --gradient_accumulation_steps 2 --train_set_path $dataset_path --output_dir $m_model_dir --hub_model_id $m_hub_id
    conda deactivate
    break
done