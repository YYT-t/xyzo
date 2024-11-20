huggingface-cli login  --token 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

git clone https://github.com/YYT-t/xyzo.git
cd xyzo

conda env create -f environment.yaml
conda activate yy

ACCELERATE_LOG_LEVEL=info accelerate launch --main_process_port $PORT1 z_sa_7.py \
    --model_name google/gemma-2-2b-it  \
    --train_set_path openai/gsm8k \
    --deepspeed ./deepspeed_configs/deepspeed_3.json \
    --output_path ./Q_models/tt_7_gemma \
    --max_length 256 \
    --save_every_steps 50 \
    --per_device_train_batch_size 4

huggingface-cli upload YYT-t/gemma-2-2b-it-e-7 Q_models/tt_7_gemma/debug2 --token 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