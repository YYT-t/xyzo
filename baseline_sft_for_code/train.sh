
export WANDB_API_KEY=84f03efa3815c8727157b1951519ce4b0f2a190a
wandb login --relogin ${WANDB_API_KEY}
huggingface-cli login  --token =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
git clone https://github.com/YYT-t/xyzo.git
cd xyzo
cd baseline_sft_for_code
conda activate sft || conda env create -f environment.yml && conda activate sft
export HF_TOKEN=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
TASK_ID=code_opencoder_edu
MODEL_NAME=deepseek-ai/deepseek-coder-6.7b-instruct
MODEL_SAVE_NAME=baseline-deepseek-coder-6.7b-instruct-sft
DATASET_NAME=OpenCoder-LLM/opc-sft-stage2

ACCELERATE_LOG_LEVEL=info accelerate launch --main_process_port  $PORT1 --num_processes 8 baseline_sft.py --deepspeed deepspeed_configs/deepspeed_3.json  --model_name ${MODEL_NAME} --output_dir ./${MODEL_SAVE_NAME} --hub_model_id ${MODEL_SAVE_NAME} --learning_rate 5e-5 --num_train_epochs 1 --Task_Type ${TASK_ID} --per_device_train_batch_size 16 --gradient_accumulation_steps 4 --wandb_project EM

#bash eval.sh ./${MODEL_SAVE_NAME}
