# conda env create -f environment.yaml
# conda activate math_eval

# cd latex2sympy
# pip install -e .
# cd ..
# pip install -r requirements.txt 
# pip install vllm==0.5.1 --no-build-isolation
# pip install transformers==4.42.3
export HF_TOKEN=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
export CUDA_VISIBLE_DEVICES="1,3"
PROMPT_TYPE="llama-3-8b-it"
MODEL_NAME_OR_PATH="/opt/tiger/xyzo/Meta-Llama-3-8B-Instruct-metamath_sample_1000_tp/sft_model_zq_raw"
bash sh/my_eval.sh $PROMPT_TYPE $MODEL_NAME_OR_PATH