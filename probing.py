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
    Trainer,
    TrainingArguments,
)
from transformers.utils import PaddingStrategy
import wandb
import sys
import os
from utils import regularized_logp_tensor
import lm_eval
import subprocess

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
    use_template: Optional[bool] = field(default=False)
    label_smoothing: Optional[float] = field(default=0.0)
    model_path: Optional[str] = field(default="None")
    save_strategy: Optional[str] = field(default="steps")
    wandb_project: Optional[str] = field(default="E_step_ent")
    upload_to_hub: Optional[bool] = field(default=True)



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



model = AutoModelForCausalLM.from_pretrained(
    script_args.model_name, torch_dtype=torch.bfloat16, use_flash_attention_2=False
)
our_base_model = AutoModelForCausalLM.from_pretrained(
    script_args.model_name, torch_dtype=torch.bfloat16, use_flash_attention_2=False
)

model.config.use_cache = not script_args.gradient_checkpointing


test_str = "Answer the question based on the following example:\nQuestion: Samantha’s last name has three fewer letters than Bobbie’s last name. If Bobbie took two letters off her last name, she would have a last name twice the length of Jamie’s. Jamie’s full name is Jamie Grey. How many letters are in Samantha’s last name? Answer: There are 4 letters in Jamie’s last name, so Bobbie’s name is 4*2 +2 = 10 letters long. Samantha’s last name is 3 letters shorter than Bobbie’s, so there are 10 - 3 = 7 letters in Samantha’s last name.\nQuestion: Diane bakes four trays with 25 gingerbreads in each tray and three trays with 20 gingerbreads in each tray. How many gingerbreads does Diane bake?"

input_ids = tokenizer(test_str, return_tensors="pt")["input_ids"].to(model.device)
test_rational = model.generate(input_ids=input_ids, max_new_tokens=5,\
                                       tokenizer=tokenizer,do_sample=False)
logits = model(input_ids=input_ids, return_dict=True).logits
print("test_rational logits=", logits)
arg_max_logits = torch.argmax(logits[0, :, :], dim=-1)
print("arg_max_logits=", arg_max_logits)
print("arg_max_logits_decode=", tokenizer.decode(arg_max_logits))

print("last_max=", tokenizer.decode(torch.argmax(logits[0, -1, :]).unsqueeze(0)))
print("max_value=", torch.max(logits[0, -1, :]))

print(tokenizer.decode(test_rational[0], skip_special_tokens=False))
