export HF_TOKEN=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
CUDA_VISIBLE_DEVICES="2" lm_eval \
        --model hf \
        --model_args pretrained=/projects/p32658/xyzo/Meta-Llama-3-8B-Instruct-metamath_sample_1000_tp/m-iter-1_zq_raw \
        --tasks gsm8k \
        --device cuda:0 \
        --batch_size 8 \
        --output_path ./Logs \
        --log_samples \
        --num_fewshot 0\
        --apply_chat_template