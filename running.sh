huggingface-cli login  --token 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

git clone https://github.com/YYT-t/xyzo.git
cd xyzo

conda env create -f environment.yaml
conda activate yy

ACCELERATE_LOG_LEVEL=info accelerate launch --main_process_port $PORT1 E_step_ent.py \
    --model_name google/gemma-2-9b-it  \
    --train_set_path openai/gsm8k \
    --deepspeed ./deepspeed_configs/deepspeed_3.json \
    --output_suffix "" \
    --ent_coeff 0.05 \
    --num_beams 1\
    --do_sample False \
    --temperature 0.8 \
    --max_length 256 \
    --save_every_steps 50 \
    --gradient_accumulation_steps 1 \
    --per_device_train_batch_size 16 \