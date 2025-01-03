# source ~/.bashrc
export CUDA_VISIBLE_DEVICES="0,1,2,3"
#export WANDB_API_KEY=ee43df2d6680a9ce636f698eba4b5534c4336452
#export HF_TOKEN=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
#huggingface-cli login  --token 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
#git clone https://github.com/YYT-t/xyzo.git
#cd xyzo
#conda env create -f environment.yaml
#pip install flash-attn==2.6.3
#conda env create -f environment_sft.yaml

#model_name="Mistral-7B-Instruct-v0.2"
#company="mistralai"

company="meta-llama"
model_name="Meta-Llama-3-8B-Instruct"
critic_model_name="${company}/${model_name}"
task_pre="math"
task_suf="gsm"
# conda init bash
num_samples=7000
path="./${model_name}-${task_suf}_sample_${num_samples}_tp"
export HF_TOKEN=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

conda activate yy
mkdir $path

sft_model_dir="${path}/sft_model"
sft_hub_id="${task_pre}_${task_suf}-${model_name}-sft-sample_${num_samples}_tp"
dataset_path="ZhangShenao/sft-${task_pre}_${task_suf}-${model_name}-iter${i}_sample_${num_samples}_tp"
split="[:${num_samples}]"

python inference.py --model_path "${company}/${model_name}" --task_type "${task_pre}_${task_suf}" --dataset_path $dataset_path --iter $i --dataset_fraction $split --for_sft || exit 1
# accelerate launch  m_sft.py --deepspeed deepspeed_configs/deepspeed_3.json --num_train_epochs 3 --model_name $e_input_model  --per_device_train_batch_size 8 --gradient_accumulation_steps 4 --train_set_path $dataset_path --output_dir $m_model_dir --hub_model_id $m_hub_id || exit 1
# huggingface-cli upload "ZhangShenao/${m_hub_id}_msft" "${m_model_dir}/final_checkpoint" --token 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
python inference_xiaojun.py --model_path "${company}/${model_name}" --dataset_path $dataset_path --save_prefix $sft_model_dir --sft_data_type zq_raw --train_step $num_samples  --task_type  "${task_pre}_${task_suf}" || exit 1
huggingface-cli upload "ZhangShenao/$sft_hub_id" "${sft_model_dir}_zq_raw" --token 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
