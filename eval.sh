# lm_eval --model hf \
#     --model_args pretrained=Q_models/tt_3/debug2\
#     --tasks gsm8k \
#     --device cuda:0\
#     --batch_size 8

huggingface-cli login --token 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


lm_eval --model hf \
    --model_args pretrained=YYT-t/gemma-2-9b-it_gsm8k_ent0.05_beam5_dosampleFalse_temp0.8_estep__final \
    --tasks gsm8k \
    --device cuda:2 \
    --batch_size 8 \
    --output_path ./Logs \
    --log_samples \