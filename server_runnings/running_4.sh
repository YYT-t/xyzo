huggingface-cli login  --token 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

git clone https://github.com/YYT-t/xyzo.git
cd xyzo

conda env create -f environment.yaml
conda activate yy

ACCELERATE_LOG_LEVEL=info accelerate launch --main_process_port $PORT1 E_step_ent_metamath.py \
    --model_name google/gemma-1.1-7b-it  \
    --train_set_path meta-math/MetaMathQA \
    --deepspeed ./deepspeed_configs/deepspeed_3.json \
    --output_suffix "" \
    --ent_coeff 1.0 \
    --num_beams 1\
    --do_sample True \
    --temperature 0.8 \
    --num_train_epochs 3 \
    --max_length 256 \
    --save_every_steps 50 \
    --gradient_accumulation_steps 1 \
    --per_device_train_batch_size 16 \