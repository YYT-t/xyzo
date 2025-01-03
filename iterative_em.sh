source ~/.bashrc

#export WANDB_API_KEY=ee43df2d6680a9ce636f698eba4b5534c4336452
#export HF_TOKEN=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
#huggingface-cli login  --token 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
#git clone https://github.com/YYT-t/xyzo.git
#cd xyzo
#conda env create -f environment.yaml
#pip install flash-attn==2.6.3
#conda env create -f environment_sft.yaml

iter_num=3

#model_name="Mistral-7B-Instruct-v0.2"
#company="mistralai"

company="google"
model_name="gemma-2-9b-it"
task_pre="math"
task_suf="metamath"
conda init bash
path="./${model_name}-${task_suf}"
num_samples=4000
export HF_TOKEN=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
for i in $(seq 1 $iter_num); do
    conda activate yy
    mkdir $path
    e_input_model="${path}/m-iter-$((i-1))_zq_raw"
    e_model_dir="${path}/e-iter-$i"
    m_model_dir="${path}/m-iter-$i"
    m_hub_id="${model_name}-m-iter-$i"
    dataset_path="ZhangShenao/new-${model_name}-iter$i"
    if [ "$i" -eq 1 ]; then
        e_input_model="${company}/${model_name}"
    else
        echo "iteration $i"
    fi
    if [ "$i" -eq 1 ]; then
        split="[:4000]"
    elif [ "$i" -eq 2 ]; then
        split="[4000:8000]"
    else
        split="[8000:12000]"
    fi
    python xiaojun_E_step_ent_PPO.py --model_name $e_input_model --critic_model_name "google/gemma-2-2b-it" --task_type "${task_pre}_${task_suf}${split}" --model_path $e_model_dir || exit 1
    python inference.py --model_path "${e_model_dir}/final_checkpoint" --task_type "${task_pre}_${task_suf}" --dataset_path $dataset_path --iter $i || exit 1
#    conda deactivate
#    conda activate sft_debug
    python inference_xiaojun.py --model_path $e_input_model --dataset_path $dataset_path --save_prefix $m_model_dir --sft_data_type zq_raw --train_step $num_samples || exit 1
    huggingface-cli upload "ZhangShenao/$m_hub_id" "${m_model_dir}_zq_raw" --token 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
   # accelerate launch m_sft.py --deepspeed deepspeed_configs/deepspeed_3.json --model_name $e_input_model --attn_implementation eager --per_device_train_batch_size 4 --gradient_accumulation_steps 4 --train_set_path $dataset_path --output_dir $m_model_dir --hub_model_id $m_hub_id || exit 1
#    conda deactivate
done
