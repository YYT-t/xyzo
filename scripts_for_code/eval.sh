conda activate eval_code || conda create --name eval_code python=3.11 && conda activate eval_code
pip install huggingface
huggingface-cli login --token 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

pip install --upgrade "evalplus[vllm] @ git+https://github.com/evalplus/evalplus"
# Or `pip install "evalplus[vllm]" --upgrade` for the latest stable release


MODEL=$1
CUDA_VISIBLE_DEVICES=0 evalplus.evaluate --model ${MODEL} \
                  --dataset [humaneval|mbpp]             \
                  --backend vllm                         \
                  --greedy