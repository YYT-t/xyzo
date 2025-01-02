from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

# import evaluate
import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset
import deepspeed
from copy import deepcopy
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    PreTrainedTokenizerBase,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    GPT2Tokenizer, GPT2LMHeadModel,
    GemmaForCausalLM
)
from transformers.utils import PaddingStrategy
from transformers.generation.stopping_criteria import StopStringCriteria
import pdb
from collators import MyDataCollator

instruct_prompt = r"Answer the question based on the following example:"
example1 = r"""Question: Jack is stranded on a desert island. He wants some salt to season his fish. He collects 2 liters of seawater in an old bucket. If the water is 20% salt, how many ml of salt will Jack get when all the water evaporates? Answer: First find how many liters of the seawater are salt: 2 liters * 20% = 0.4 liters Then multiply that amount by 1000 ml/liter to find the number of ml of salt Jack gets: 0.4 liters * 1000 ml/liter = 400 ml."""
example2 = r"""Question: Samantha’s last name has three fewer letters than Bobbie’s last name. If Bobbie took two letters off her last name, she would have a last name twice the length of Jamie’s. Jamie’s full name is Jamie Grey. How many letters are in Samantha’s last name? Answer: There are 4 letters in Jamie’s last name, so Bobbie’s name is 4*2 +2 = 10 letters long. Samantha’s last name is 3 letters shorter than Bobbie’s, so there are 10 - 3 = 7 letters in Samantha’s last name."""
#few_shot_cot_prompt = instruct_prompt + '\n' + example2 + f'\nQuestion: '  #'\n' + example1
few_shot_cot_prompt = ''
# Define and parse arguments.
@dataclass
class ScriptArguments:
    """
    These arguments vary depending on how many GPUs you have, what their capacity and features are, and what size model you want to train.
    """
    local_rank: Optional[int] = field(
        default=-1, metadata={"help": "Used for multi-gpu"})

    deepspeed: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to deepspeed config if using deepspeed. You may need this if the model that you want to train doesn't fit on a single GPU."
        },
    )
    per_device_train_batch_size: Optional[int] = field(default=8)
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
    train_set_path: Optional[str] = field(
        default="openai/gsm8k",
        metadata={"help": "The dir of the subset of the training data to use"},
    )
    output_path: Optional[str] = field(
        default="./Q_models/debug",
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

    save_every_steps: Optional[int] = field(
        default=999999,
        metadata={"help": "Save the model every x steps"},
    )
    eval_every_steps: Optional[int] = field(
        default=999999,
        metadata={"help": "Eval the model every x steps"},
    )
    lora_alpha: Optional[float] = field(default=16, metadata={"help": "the lora alpha parameter"})
    lora_dropout: Optional[float] = field(default=0.05, metadata={"help": "the lora dropout parameter"})
    lora_r: Optional[int] = field(default=8, metadata={"help": "the lora r parameter"})


parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]

tokenizer_name = script_args.model_name
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name) #AutoTokenizer

tokenizer.model_max_length = script_args.max_length
#tokenizer.pad_token = tokenizer.eos_token
#tokenizer.pad_token_id = tokenizer.eos_token_id
#tokenizer.add_special_tokens({'sep_token': '[SEP]'})
# Adjusted according to the base model
# Need to do this for the models that don't have an official pad token.
tokenizer.truncation_side = "left"
tokenizer.padding_side = "left"
tokenizer.pad_token = tokenizer.eos_token
print("pad_token_id: ", tokenizer.pad_token_id)
print("eos_token_id: ", tokenizer.eos_token_id)
print("pad_token: ", tokenizer.pad_token)
print("eos_token: ", tokenizer.eos_token)
print("decode 0: ", tokenizer.decode([0]))
print("decode 50256: ", tokenizer.decode([50256]))
#tokenizer.model_max_length = script_args.max_length

# Get the dataset
train_path = script_args.train_set_path
output_name = script_args.output_path


def tokenize(sample):
   # sample["question"] = sample["question"]
    # print("sample=", sample)
    tokenized = tokenizer(sample['question'],truncation=True)
    sample["input_ids"] = tokenized["input_ids"]
    sample["attention_mask"] = tokenized["attention_mask"]
    labels = tokenizer(sample['answer'], truncation=True)
    sample["labels"] = labels["input_ids"]
    return sample

train_dataset = load_dataset(train_path, "main")["train"] #.shuffle(seed=42)
train_dataset = train_dataset.map(tokenize, num_proc=16, remove_columns=["question", "answer"])

maxi = 0
for i in range(len(train_dataset["input_ids"])):
    maxi = max(maxi, len(train_dataset["input_ids"][i]))
    print(maxi)
print(maxi)