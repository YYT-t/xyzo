# lm_eval --model hf \
#     --model_args pretrained=Q_models/tt_3/debug2\
#     --tasks gsm8k \
#     --device cuda:0\
#     --batch_size 8

huggingface-cli login --token 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


lm_eval --model hf \
--model_args pretrained=Q_models/tt_4_llama/debug2 \
--tasks gsm8k \
--device cuda:0 \
--batch_size 8 \
--output_path ./Logs \
--log_samples \