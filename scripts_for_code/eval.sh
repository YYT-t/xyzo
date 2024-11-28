# lm_eval --model hf \
#     --model_args pretrained=Q_models/tt_3/debug2\
#     --tasks gsm8k \
#     --device cuda:0\
#     --batch_size 8

huggingface-cli login --token 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

pip install --upgrade "evalplus[vllm] @ git+https://github.com/evalplus/evalplus"
# Or `pip install "evalplus[vllm]" --upgrade` for the latest stable release


MODEL=xxx
CUDA_VISIBLE_DEVICES=0 evalplus.evaluate --model YYT-t/${MODEL} \
                  --dataset [humaneval|mbpp]             \
                  --backend vllm                         \
                  --greedy
