export HF_TOKEN=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

# Do M-step with Q-generated data.
mlx worker launch --type=a100-80g -- python inference_xiaojun.py --model_path xiaojun_out/PPO_ansprob/final_checkpoint/ --dataset_path YYT-t/iterative-gsm-iter1 --save_prefix ./xiaojun_out/Mstep_v0 --sft_data_type zq_raw --train_step 2000
# Do M-step with ground truth data.
mlx worker launch --type=a100-80g -- python inference_xiaojun.py --model_path xiaojun_out/PPO_ansprob/final_checkpoint/ --dataset_path YYT-t/iterative-gsm-iter1 --save_prefix ./xiaojun_out/Mstep_v0 --sft_data_type zgt_raw --train_step 2000

# Evaluate the model SFT'ed on Q data
mlx worker launch --gpu=1 --type=a100-80g -- lm_eval --model hf --model_args pretrained=./xiaojun_out/Mstep_v0_zq_raw --tasks gsm8k --device cuda:0 --batch_size 8 --output_path ./Logs --log_samples --num_fewshot 0
# Evaluate the model SFT'ed on gt data
mlx worker launch --gpu=1 --type=a100-80g -- lm_eval --model hf --model_args pretrained=./xiaojun_out/Mstep_v0_zgt_raw --tasks gsm8k --device cuda:0 --batch_size 8 --output_path ./Logs --log_samples --num_fewshot 0
# Evaluate the pretrained model
mlx worker launch --gpu=1 --type=a100-80g -- lm_eval --model hf --model_args pretrained=google/gemma-2-2b-it --tasks gsm8k --device cuda:0 --batch_size 8 --output_path ./Logs --log_samples --num_fewshot 0

# ======above: commands for replicating the better performance on gemma-2-2b
# ======below: test space=======

#mlx worker launch --type=a100-80g -- python inference_xiaojun.py --model_path xiaojun_out/PPO_ansprob/final_checkpoint/ --dataset_path YYT-t/iterative-gsm-iter1 --save_prefix ./xiaojun_out/Mstep_v0 --sft_data_type zgt_raw --with_template --train_step 2000
##mlx worker launch --type=a100-80g -- python inference_xiaojun.py --model_path xiaojun_out/PPO_ansprob/final_checkpoint/ --dataset_path YYT-t/iterative-gsm-iter1 --save_prefix ./xiaojun_out/Mstep_v0 --sft_data_type zgt_raw --train_step 2000
##mlx worker launch --type=a100-80g -- python inference_xiaojun.py --model_path xiaojun_out/PPO_ansprob/final_checkpoint/ --dataset_path YYT-t/iterative-gsm-iter1 --save_prefix ./xiaojun_out/Mstep_v0 --sft_data_type zgt_filter --train_step 2000
###mlx worker launch --type=a100-80g -- python inference_xiaojun.py --model_path xiaojun_out/PPO_ansprob/final_checkpoint/ --dataset_path YYT-t/iterative-gsm-iter1 --save_prefix ./xiaojun_out/Mstep_v0 --sft_data_type zq_filter --train_step 2000
###mlx worker launch --type=a100-80g -- python inference_xiaojun.py --model_path xiaojun_out/PPO_ansprob/final_checkpoint/ --dataset_path YYT-t/iterative-gsm-iter1 --save_prefix ./xiaojun_out/Mstep_v0 --sft_data_type zq_raw --train_step 2000

#mlx worker launch --gpu=1 --type=a100-80g -- lm_eval --model hf --model_args pretrained=./xiaojun_out/Mstep_v0_zgt_raw --tasks gsm8k --device cuda:0 --batch_size 8 --output_path ./Logs --log_samples
#mlx worker launch --gpu=1 --type=a100-80g -- lm_eval --model hf --model_args pretrained=./xiaojun_out/Mstep_v0_zgt_filter --tasks gsm8k --device cuda:0 --batch_size 8 --output_path ./Logs --log_samples
#mlx worker launch --gpu=1 --type=a100-80g -- lm_eval --model hf --model_args pretrained=./xiaojun_out/Mstep_v0_zq_raw --tasks gsm8k --device cuda:0 --batch_size 8 --output_path ./Logs --log_samples
#mlx worker launch --gpu=1 --type=a100-80g -- lm_eval --model hf --model_args pretrained=./xiaojun_out/Mstep_v0_zq_filter --tasks gsm8k --device cuda:0 --batch_size 8 --output_path ./Logs --log_samples
#mlx worker launch --gpu=1 --type=a100-80g -- lm_eval --model hf --model_args pretrained=google/gemma-2-2b-it --tasks gsm8k --device cuda:0 --batch_size 8 --output_path ./Logs --log_samples
#mlx worker launch --gpu=1 --type=a100-80g -- lm_eval --model hf --model_args pretrained=google/gemma-2-2b-it --tasks gsm8k --device cuda:0 --batch_size 8 --output_path ./Logs --log_samples --num_fewshot 0
#mlx worker launch --gpu=1 --type=a100-80g -- lm_eval --model hf --model_args pretrained=google/gemma-2-2b-it --tasks gsm8k --device cuda:0 --batch_size 8 --output_path ./Logs --log_samples --apply_chat_template



#mlx worker launch --type=a100-80g -- python inference_xiaojun.py --model_path YYT-t/PPO_ansprob_9b_final_checkpoint --dataset_path YYT-t/iterative-gsm-iter1 --save_prefix ./xiaojun_out/Mstep_9b_v0 --sft_data_type zgt_filter --train_step 2000
#mlx worker launch --type=a100-80g -- python inference_xiaojun.py --model_path YYT-t/PPO_ansprob_9b_final_checkpoint --dataset_path YYT-t/iterative-gsm-iter1 --save_prefix ./xiaojun_out/Mstep_9b_v0 --sft_data_type zq_filter --train_step 2000
#mlx worker launch --type=a100-80g -- python inference_xiaojun.py --model_path YYT-t/PPO_ansprob_9b_final_checkpoint --dataset_path YYT-t/iterative-gsm-iter1 --save_prefix ./xiaojun_out/Mstep_9b_v0 --sft_data_type zgt_raw --train_step 2000
#mlx worker launch --type=a100-80g -- python inference_xiaojun.py --model_path YYT-t/PPO_ansprob_9b_final_checkpoint --dataset_path YYT-t/iterative-gsm-iter1 --save_prefix ./xiaojun_out/Mstep_9b_v0 --sft_data_type zq_raw --train_step 2000
#
#mlx worker launch --gpu=1 --type=a100-80g -- lm_eval --model hf --model_args pretrained=./xiaojun_out/Mstep_9b_v0_zgt_raw --tasks gsm8k --device cuda:0 --batch_size 8 --output_path ./Logs --log_samples
#mlx worker launch --gpu=1 --type=a100-80g -- lm_eval --model hf --model_args pretrained=./xiaojun_out/Mstep_9b_v0_zgt_filter --tasks gsm8k --device cuda:0 --batch_size 8 --output_path ./Logs --log_samples
#mlx worker launch --gpu=1 --type=a100-80g -- lm_eval --model hf --model_args pretrained=./xiaojun_out/Mstep_9b_v0_zq_raw --tasks gsm8k --device cuda:0 --batch_size 8 --output_path ./Logs --log_samples
#mlx worker launch --gpu=1 --type=a100-80g -- lm_eval --model hf --model_args pretrained=./xiaojun_out/Mstep_9b_v0_zq_filter --tasks gsm8k --device cuda:0 --batch_size 8 --output_path ./Logs --log_samples
#mlx worker launch --gpu=1 --type=a100-80g -- lm_eval --model hf --model_args pretrained=google/gemma-2-9b-it --tasks gsm8k --device cuda:0 --batch_size 8 --output_path ./Logs --log_samples


# zero-shot and template
#mlx worker launch --gpu=1 --type=a100-80g -- lm_eval --model hf --model_args pretrained=google/gemma-2-2b-it --tasks gsm8k --device cuda:0 --batch_size 8 --output_path ./Logs --log_samples --num_fewshot 0
#mlx worker launch --gpu=1 --type=a100-80g -- lm_eval --model hf --model_args pretrained=./xiaojun_out/Mstep_v0_zgt_raw --tasks gsm8k --device cuda:0 --batch_size 8 --output_path ./Logs --log_samples --num_fewshot 0
#mlx worker launch --gpu=1 --type=a100-80g -- lm_eval --model hf --model_args pretrained=./xiaojun_out/Mstep_v0_zgt_filter --tasks gsm8k --device cuda:0 --batch_size 8 --output_path ./Logs --log_samples --num_fewshot 0
#mlx worker launch --gpu=1 --type=a100-80g -- lm_eval --model hf --model_args pretrained=./xiaojun_out/Mstep_v0_zq_raw --tasks gsm8k --device cuda:0 --batch_size 8 --output_path ./Logs --log_samples --num_fewshot 0
#mlx worker launch --gpu=1 --type=a100-80g -- lm_eval --model hf --model_args pretrained=./xiaojun_out/Mstep_v0_zq_filter --tasks gsm8k --device cuda:0 --batch_size 8 --output_path ./Logs --log_samples --num_fewshot 0
#
#mlx worker launch --type=a100-80g -- python inference_xiaojun.py --model_path xiaojun_out/PPO_ansprob/final_checkpoint/ --dataset_path YYT-t/iterative-gsm-iter1 --save_prefix ./xiaojun_out/Mstep_v0 --sft_data_type zgt_raw --with_template --train_step 2000
#mlx worker launch --type=a100-80g -- python inference_xiaojun.py --model_path xiaojun_out/PPO_ansprob/final_checkpoint/ --dataset_path YYT-t/iterative-gsm-iter1 --save_prefix ./xiaojun_out/Mstep_v0 --sft_data_type zgt_filter --with_template --train_step 2000
#mlx worker launch --type=a100-80g -- python inference_xiaojun.py --model_path xiaojun_out/PPO_ansprob/final_checkpoint/ --dataset_path YYT-t/iterative-gsm-iter1 --save_prefix ./xiaojun_out/Mstep_v0 --sft_data_type zq_raw --with_template --train_step 2000
#mlx worker launch --type=a100-80g -- python inference_xiaojun.py --model_path xiaojun_out/PPO_ansprob/final_checkpoint/ --dataset_path YYT-t/iterative-gsm-iter1 --save_prefix ./xiaojun_out/Mstep_v0 --sft_data_type zq_filter --with_template --train_step 2000
#mlx worker launch --gpu=1 --type=a100-80g -- lm_eval --model hf --model_args pretrained=google/gemma-2-2b-it --tasks gsm8k --device cuda:0 --batch_size 8 --output_path ./Logs --log_samples --num_fewshot 0 --apply_chat_template
#mlx worker launch --gpu=1 --type=a100-80g -- lm_eval --model hf --model_args pretrained=./xiaojun_out/Mstep_v0_zgt_raw_withtemplate --tasks gsm8k --device cuda:0 --batch_size 8 --output_path ./Logs --log_samples --num_fewshot 0 --apply_chat_template
#mlx worker launch --gpu=1 --type=a100-80g -- lm_eval --model hf --model_args pretrained=./xiaojun_out/Mstep_v0_zgt_filter_withtemplate --tasks gsm8k --device cuda:0 --batch_size 8 --output_path ./Logs --log_samples --num_fewshot 0 --apply_chat_template
#mlx worker launch --gpu=1 --type=a100-80g -- lm_eval --model hf --model_args pretrained=./xiaojun_out/Mstep_v0_zq_raw_withtemplate --tasks gsm8k --device cuda:0 --batch_size 8 --output_path ./Logs --log_samples --num_fewshot 0 --apply_chat_template
#mlx worker launch --gpu=1 --type=a100-80g -- lm_eval --model hf --model_args pretrained=./xiaojun_out/Mstep_v0_zq_filter_withtemplate --tasks gsm8k --device cuda:0 --batch_size 8 --output_path ./Logs --log_samples --num_fewshot 0 --apply_chat_template

##============== 9b ================
#mlx worker launch --gpu=1 --type=a100-80g -- lm_eval --model hf --model_args pretrained=google/gemma-2-9b-it --tasks gsm8k --device cuda:0 --batch_size 8 --output_path ./Logs --log_samples --num_fewshot 0
#mlx worker launch --gpu=1 --type=a100-80g -- lm_eval --model hf --model_args pretrained=./xiaojun_out/Mstep_9b_v0_zq_raw --tasks gsm8k --device cuda:0 --batch_size 8 --output_path ./Logs --log_samples --num_fewshot 0
#mlx worker launch --gpu=1 --type=a100-80g -- lm_eval --model hf --model_args pretrained=./xiaojun_out/Mstep_9b_v0_zgt_raw --tasks gsm8k --device cuda:0 --batch_size 8 --output_path ./Logs --log_samples --num_fewshot 0
#mlx worker launch --gpu=1 --type=a100-80g -- lm_eval --model hf --model_args pretrained=./xiaojun_out/Mstep_9b_v0_zq_filter --tasks gsm8k --device cuda:0 --batch_size 8 --output_path ./Logs --log_samples --num_fewshot 0
#mlx worker launch --gpu=1 --type=a100-80g -- lm_eval --model hf --model_args pretrained=./xiaojun_out/Mstep_9b_v0_zgt_filter --tasks gsm8k --device cuda:0 --batch_size 8 --output_path ./Logs --log_samples --num_fewshot 0
#
#mlx worker launch --type=a100-80g -- python inference_xiaojun.py --model_path YYT-t/PPO_ansprob_9b_final_checkpoint --dataset_path YYT-t/iterative-gsm-iter1 --save_prefix ./xiaojun_out/Mstep_9b_v0 --sft_data_type zgt_raw --with_template --train_step 2000
#mlx worker launch --type=a100-80g -- python inference_xiaojun.py --model_path YYT-t/PPO_ansprob_9b_final_checkpoint --dataset_path YYT-t/iterative-gsm-iter1 --save_prefix ./xiaojun_out/Mstep_9b_v0 --sft_data_type zgt_filter --with_template --train_step 2000
#mlx worker launch --type=a100-80g -- python inference_xiaojun.py --model_path YYT-t/PPO_ansprob_9b_final_checkpoint --dataset_path YYT-t/iterative-gsm-iter1 --save_prefix ./xiaojun_out/Mstep_9b_v0 --sft_data_type zq_raw --with_template --train_step 2000
#mlx worker launch --type=a100-80g -- python inference_xiaojun.py --model_path YYT-t/PPO_ansprob_9b_final_checkpoint --dataset_path YYT-t/iterative-gsm-iter1 --save_prefix ./xiaojun_out/Mstep_9b_v0 --sft_data_type zq_filter --with_template --train_step 2000
#
#mlx worker launch --gpu=1 --type=a100-80g -- lm_eval --model hf --model_args pretrained=google/gemma-2-9b-it --tasks gsm8k --device cuda:0 --batch_size 8 --output_path ./Logs --log_samples --num_fewshot 0 --apply_chat_template
#mlx worker launch --gpu=1 --type=a100-80g -- lm_eval --model hf --model_args pretrained=./xiaojun_out/Mstep_9b_v0_zq_raw_withtemplate --tasks gsm8k --device cuda:0 --batch_size 8 --output_path ./Logs --log_samples --num_fewshot 0 --apply_chat_template
#mlx worker launch --gpu=1 --type=a100-80g -- lm_eval --model hf --model_args pretrained=./xiaojun_out/Mstep_9b_v0_zgt_raw_withtemplate --tasks gsm8k --device cuda:0 --batch_size 8 --output_path ./Logs --log_samples --num_fewshot 0 --apply_chat_template
#mlx worker launch --gpu=1 --type=a100-80g -- lm_eval --model hf --model_args pretrained=./xiaojun_out/Mstep_9b_v0_zq_filter_withtemplate --tasks gsm8k --device cuda:0 --batch_size 8 --output_path ./Logs --log_samples --num_fewshot 0 --apply_chat_template
#mlx worker launch --gpu=1 --type=a100-80g -- lm_eval --model hf --model_args pretrained=./xiaojun_out/Mstep_9b_v0_zgt_filter_withtemplate --tasks gsm8k --device cuda:0 --batch_size 8 --output_path ./Logs --log_samples --num_fewshot 0 --apply_chat_template
