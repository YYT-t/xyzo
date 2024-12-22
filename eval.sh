# lm_eval --model hf \
#     --model_args pretrained=Q_models/tt_3/debug2\
#     --tasks gsm8k \
#     --device cuda:0\
#     --batch_size 8

huggingface-cli login --token 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


    #--model_args pretrained=./xiaojun_out/PPO_ansprob/final_checkpoint\
    #--model_args pretrained=YYT-t/PPO_ansprob_9b_final_checkpoint\
mlx worker launch --gpu=1 --type=a100-80g -- lm_eval --model hf \
    --model_args pretrained=google/gemma-2-9b-it\
	--apply_chat_template \
    --tasks gsm8k \
    --device cuda:0 \
    --batch_size 8 \
    --output_path ./Logs \
    --log_samples \
