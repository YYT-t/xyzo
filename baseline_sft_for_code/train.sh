conda activate sft || conda env create -f environment.yml && conda activate sft

TASK_ID=code_opencoder_edu
MODEL_NAME=deepseek-ai/deepseek-coder-6.7b-instruct
MODEL_SAVE_NAME=baseline-deepseek-coder-6.7b-instruct-sft
DATASET_NAME=OpenCoder-LLM/opc-sft-stage2

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ACCELERATE_LOG_LEVEL=info accelerate launch --num_processes 8 baseline_sft.py --deepspeed deepspeed_configs/deepspeed_2.json --model_name ${MODEL_NAME} --output_dir ./${MODEL_SAVE_NAME} --hub_model_id ${MODEL_SAVE_NAME} --learning_rate 5e-5 --num_train_epochs 1 --Task_Type ${TASK_ID} --per_device_train_batch_size 16 --gradient_accumulation_steps 4 

#bash eval.sh ${MODEL_SAVE_NAME}