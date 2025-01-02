from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from task_configs import task_config_check, task_data_set
# import evaluate
import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset
import deepspeed
from copy import deepcopy
# from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    PreTrainedTokenizerBase,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    GPT2Tokenizer, GPT2LMHeadModel,
    Gemma2ForCausalLM
)
from transformers.utils import PaddingStrategy
import wandb
import sys
import os
from utils import regularized_logp

import logging
@dataclass
class ScriptArguments:
    """
    These arguments vary depending on how many GPUs you have, what their capacity and features are, and what size model you want to train.
    """
    local_rank: Optional[int] = field(
        default=-1, metadata={"help": "Used for multi-gpu"})

    deepspeed: Optional[str] = field(
        # default="dp3.json",
        default=None,
        metadata={
            "help": "Path to deepspeed config if using deepspeed. You may need this if the model that you want to train doesn't fit on a single GPU."
        },
    )
    per_device_train_batch_size: Optional[int] = field(default=4)
    per_device_eval_batch_size: Optional[int] = field(default=1)
    gradient_accumulation_steps: Optional[int] = field(default=4)
    learning_rate: Optional[float] = field(default=5e-7)
    weight_decay: Optional[float] = field(default=0.001)
    model_name: Optional[str] = field(
        default="google/gemma-2b-it",  # "mistralai/Mistral-7B-Instruct-v0.2",
        metadata={
            "help": "The model that you want to train from the Hugging Face hub. E.g. gpt2, gpt2-xl, bert, etc."
        },
    )
    bf16: Optional[bool] = field(
        default=True,
        metadata={
            "help": "This essentially cuts the training time in half if you want to sacrifice a little precision and have a supported GPU."
        },
    )
    num_train_epochs: Optional[int] = field(
        default=1,
        metadata={"help": "The number of training epochs for the reward model."},
    )
    output_suffix: Optional[str] = field(
        default="",
        metadata={"help": "The dir for output model"},
    )
    gradient_checkpointing: Optional[bool] = field(
        default=True,
        metadata={"help": "Enables gradient checkpointing."},
    )
    optim: Optional[str] = field(
        default="adamw_torch",
        metadata={"help": "The optimizer to use."},
    )
    lr_scheduler_type: Optional[str] = field(
        default="cosine",
        metadata={"help": "The lr scheduler"},
    )
    max_length: Optional[int] = field(default=256)
    model_max_length: Optional[int] = field(default=256)

    save_every_steps: Optional[int] = field(
        default=999999,
        metadata={"help": "Save the model every x steps"},
    )
    eval_every_steps: Optional[int] = field(
        default=999999,
        metadata={"help": "Eval the model every x steps"},
    )
    prompt_path: Optional[str] = field(
        default="prompts/math_prompt.txt",
        metadata={"help": "path to get the cot prompt"},
    )
    task_type: Optional[str] = field(
        default="math_metamath",
        metadata={"help": "math or code"},
    )
    ent_coeff: Optional[float] = field(default=0.05)
    temperature: Optional[float] = field(default=0.8)
    num_beams: Optional[int] = field(default=5)
    do_sample: Optional[bool] = field(default=True)
    log_regulaizer: Optional[bool] = field(default=False)
    regu_eps: Optional[float] = field(default=1e-6)
    model_path: Optional[str] = field(default="None")
    save_strategy: Optional[str] = field(default="steps")
    wandb_project: Optional[str] = field(default="E_step_ent")



parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]

task_config = task_config_check(script_args.task_type)
train_set_path, train_dataset = task_data_set(script_args.task_type)

tokenizer_name = script_args.model_name
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name) #AutoTokenizer

tokenizer.model_max_length = script_args.model_max_length
tokenizer.truncation_side = "left"
tokenizer.padding_side = "left"
tokenizer.pad_token = tokenizer.eos_token


base_model_name = script_args.model_name.split("/")[1]

data_name = train_set_path.split("/")[1]

trained_model_name = f"{base_model_name}_{data_name}_ent{script_args.ent_coeff}_\
beam{script_args.num_beams}_dosample{script_args.do_sample}_temp{script_args.temperature}_\
estep_{script_args.output_suffix}_totalepoch{script_args.num_train_epochs}"

if script_args.model_path == "None":
    output_name = f"./Q_models/{trained_model_name}"
else:
    output_name = script_args.model_path
    
if not os.path.exists(output_name):
    os.makedirs(output_name)
# output_name = script_args.model_path

train_dataset = train_dataset.map(task_config.tokenize_E(tokenizer), num_proc=16)
# train_dataset = train_dataset.select(range(2))


# training_args.push_to_hub = True
# training_args.hub_strategy = "every_save"

# training_args.hub_model_id = f"YYT-t/{trained_model_name}"
# training_args.hub_token = "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"
# Define the trainer
os.environ["WANDB_PROJECT"]=script_args.wandb_project
training_args = TrainingArguments(
    output_dir=output_name,
    learning_rate=script_args.learning_rate,
    per_device_train_batch_size=script_args.per_device_train_batch_size,
    per_device_eval_batch_size=script_args.per_device_eval_batch_size,
    num_train_epochs=script_args.num_train_epochs,
  #  weight_decay=script_args.weight_decay,
    evaluation_strategy="steps",
    eval_steps=script_args.eval_every_steps,
    save_strategy=script_args.save_strategy,
    save_steps=script_args.save_every_steps,
    gradient_accumulation_steps=script_args.gradient_accumulation_steps,
    gradient_checkpointing=script_args.gradient_checkpointing,
    deepspeed=script_args.deepspeed,
    local_rank=script_args.local_rank,
    #remove_unused_columns=True,
    remove_unused_columns=False,
    label_names=[],
    bf16=script_args.bf16,
    logging_strategy="steps",
    logging_steps=10,
    optim=script_args.optim,
    lr_scheduler_type=script_args.lr_scheduler_type,
    warmup_ratio=0.1,
    report_to='wandb',
    # run_name="E_step_ent"
    # push_to_hub=True,
    # hub_strategy="checkpoint",
    # hub_model_id=f"YYT-t/3",
    # hub_token="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"
)

model = AutoModelForCausalLM.from_pretrained(
    script_args.model_name, torch_dtype=torch.bfloat16, use_flash_attention_2=False
)

model.config.use_cache = not script_args.gradient_checkpointing
original_columns = train_dataset.column_names

print(train_dataset[0])