#!/bin/bash
# cd em
# conda env create -f environment.yml
# conda activate sft

export CUDA_VISIBLE_DEVICES=0,1,2,3

conda env create -f environment.yaml
conda env create -f environment_sft.yaml

export HF_TOKEN=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
iter_num=3
path="./gemma-2-2b"
for i in $(seq 1 $iter_num); do
    conda activate yy
    
    mkdir $path
    e_input_model="${path}/m-model-iter-$((i-1))"
    e_model_dir="${path}/e-model-iter-$i"
    m_model_dir="${path}/m-model-iter-$i"
    m_hub_id="gemma2-2b-it-m-model-iter-$i"
    dataset_path="YYT-t/iterative-gsm-iter$i"
    if [ "$i" -eq 1 ]; then
        e_input_model="google/gemma-2-2b-it"
    else
        echo "iteration $i"
    fi
    
    # ACCELERATE_LOG_LEVEL=info accelerate launch E_step_ent_test.py \
    # --model_name google/gemma-2-2b-it  \
    # --task_type math_gsm \
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
    # --upload_to_hub False \

    python inference_test.py --model_path "${e_model_dir}/final_checkpoint" --dataset_path $dataset_path --iter $i
    conda deactivate

    conda activate sft_debug
    accelerate launch m_sft_test.py --deepspeed deepspeed_configs/deepspeed_2.json --model_name google/gemma-2-2b-it --attn_implementation eager --per_device_train_batch_size 2 --gradient_accumulation_steps 8 --train_set_path $dataset_path --output_dir $m_model_dir --hub_model_id $m_hub_id
    conda deactivate
done