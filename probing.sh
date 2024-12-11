export WANDB_API_KEY=8b2885096bc8e2e0291c1e6aec2de6f864bba024
huggingface-cli login  --token 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

CUDA_VISIBLE_DEVICES=0,1,2,3 python probing.py \
    --model_name google/gemma-2-9b-it
