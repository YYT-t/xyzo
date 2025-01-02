export HF_TOKEN=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
export CUDA_VISIBLE_DEVICES=0,1,2,3
python inference.py --model_path "YYT-t/gemma-2-9b-it_gsm8k_ent0.05_beam5_dosampleFalse_temp0.8_estep__final" \
                --dataset_path "YYT-t/mytest" \
                --task_type "math_gsm"\
                --iter 3