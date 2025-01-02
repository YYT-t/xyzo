from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

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
    GPT2Tokenizer, GPT2LMHeadModel
)
from transformers.utils import PaddingStrategy
import pdb


instruct_prompt = r"Answer the question based on the following example:"
example1 = r"""Question: Jack is stranded on a desert island. He wants some salt to season his fish. He collects 2 liters of seawater in an old bucket. If the water is 20% salt, how many ml of salt will Jack get when all the water evaporates? Answer: First find how many liters of the seawater are salt: 2 liters * 20% = 0.4 liters Then multiply that amount by 1000 ml/liter to find the number of ml of salt Jack gets: 0.4 liters * 1000 ml/liter = 400 ml."""
example2 = r"""Question: Samantha’s last name has three fewer letters than Bobbie’s last name. If Bobbie took two letters off her last name, she would have a last name twice the length of Jamie’s. Jamie’s full name is Jamie Grey. How many letters are in Samantha’s last name? Answer: There are 4 letters in Jamie’s last name, so Bobbie’s name is 4*2 +2 = 10 letters long. Samantha’s last name is 3 letters shorter than Bobbie’s, so there are 10 - 3 = 7 letters in Samantha’s last name."""
few_shot_cot_prompt = instruct_prompt + '\n' + example2 + f'\nQuestion: '  #'\n' + example1
#few_shot_cot_prompt = ''
# Define and parse arguments.
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
    train_set_path: Optional[str] = field(
        default="openai/gsm8k",
        metadata={"help": "The dir of the subset of the training data to use"},
    )
    output_path: Optional[str] = field(
        default="./Q_models/tt_3",
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


parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]

tokenizer_name = script_args.model_name
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name) #AutoTokenizer

tokenizer.model_max_length = script_args.max_length
tokenizer.truncation_side = "left"
tokenizer.padding_side = "left"
tokenizer.pad_token = tokenizer.eos_token

# Get the dataset
train_path = script_args.train_set_path
output_name = script_args.output_path



def tokenize(sample):
    tokenized_q = tokenizer(few_shot_cot_prompt + sample['question'], truncation=True)
    answer_text = sample['answer'].split('####')[-1].strip()
    answer = f"The answer is {answer_text}."
    tokenized_a = tokenizer(answer, truncation=True)
    sample["input_ids_q"] = tokenized_q["input_ids"]
    sample["attention_mask_q"] = tokenized_q["attention_mask"]
    sample["input_ids_a"] = tokenized_a["input_ids"]
    sample["attention_mask_a"] = tokenized_a["attention_mask"]
    return sample

train_dataset = load_dataset(train_path, "main")["train"]#.shuffle(seed=42)
train_dataset = train_dataset.map(tokenize, num_proc=16)


# Define the trainer
training_args = TrainingArguments(
    output_dir=output_name,
    learning_rate=script_args.learning_rate,
    per_device_train_batch_size=script_args.per_device_train_batch_size,
    per_device_eval_batch_size=script_args.per_device_eval_batch_size,
    num_train_epochs=script_args.num_train_epochs,
  #  weight_decay=script_args.weight_decay,
    evaluation_strategy="steps",
    eval_steps=script_args.eval_every_steps,
    save_strategy="steps",
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
    report_to='wandb'
)

model = AutoModelForCausalLM.from_pretrained(
    script_args.model_name, torch_dtype=torch.bfloat16, use_flash_attention_2=False
)
our_base_model = AutoModelForCausalLM.from_pretrained(
    script_args.model_name, torch_dtype=torch.bfloat16, use_flash_attention_2=False
)

model.config.use_cache = not script_args.gradient_checkpointing
original_columns = train_dataset.column_names


def padding_func(ft_ls, padding_side, pad_token_id, return_tensors):
    max_len = max(len(ft) for ft in ft_ls)
    padded_ft_ls = []
    for ft in ft_ls:
        if padding_side == "right":
            padded_ft_ls.append(ft + [pad_token_id] * (max_len - len(ft)))
        else:
            padded_ft_ls.append([pad_token_id] * (max_len - len(ft)) + ft)
    if return_tensors == "pt":
        return torch.tensor(padded_ft_ls)

class QTrainer(Trainer):
    def __init__(self, base_model, **kwargs):
        super().__init__(**kwargs)
        if self.is_deepspeed_enabled:
            self.base_model = self._prepare_deepspeed(base_model)
        else:
            self.base_model = self.accelerator.prepare_model(base_model, evaluation_mode=True)

    def compute_loss(self, model, inputs):
        with torch.no_grad():
            inputs_ids_q_l = inputs["input_ids_q_l"]
            inputs_ids_a_r = inputs["input_ids_a_r"][:, 1:]
            mask_q_l = inputs["attention_mask_q_l"]
            rational = model.generate(input_ids=inputs_ids_q_l, attention_mask=mask_q_l, \
                                      max_new_tokens=script_args.max_length, \
                                      stop_strings="Question:", tokenizer=tokenizer)
            
            rational_decode = tokenizer.batch_decode(rational, skip_special_tokens=True)
            answer_decode = tokenizer.batch_decode(inputs_ids_a_r, skip_special_tokens=True)
            xz = []
            xz_mask = []
            xzy = []
            xzy_mask = []
            for i in range(len(rational_decode)):
                xz_tok = tokenizer(rational_decode[i] + "\n")
                xz.append(xz_tok["input_ids"])
                xz_mask.append(xz_tok["attention_mask"])
                xzy_tok = tokenizer(rational_decode[i] + "\n" + answer_decode[i])
                xzy.append(xzy_tok["input_ids"])
                xzy_mask.append(xzy_tok["attention_mask"])
            xz = padding_func(xz, "right", tokenizer.pad_token_id, "pt").to(inputs_ids_q_l.device)
            xz_mask = padding_func(xz_mask, "right", 0, "pt").to(inputs_ids_q_l.device)
            xz_labels = deepcopy(xz)
            xz_labels[xz_labels == tokenizer.pad_token_id] = -100

            xzy = padding_func(xzy, "right", tokenizer.pad_token_id, "pt").to(inputs_ids_q_l.device)
            xzy_mask = padding_func(xzy_mask, "right", 0, "pt").to(inputs_ids_q_l.device)
            xzy_labels = deepcopy(xzy)
            xzy_labels[xzy_labels == tokenizer.pad_token_id] = -100

            x = inputs["input_ids_q_r"]
            x_mask = inputs["attention_mask_q_r"]
            x_labels = deepcopy(x)
            x_labels[x_labels == tokenizer.pad_token_id] = -100
            # print("x: ", x[0])
            # print("x_mask: ", x_mask[0])
            # print("x_labels: ", x_labels[0])
            # print("xz: ", xz[0])
            # print("xz_mask: ", xz_mask[0])
            # print("xz_labels: ", xz_labels[0])
            # print("xzy: ", xzy[0])
            # print("xzy_mask: ", xzy_mask[0])
            # print("xzy_labels: ", xzy_labels[0])
            # print("decode 111:", tokenizer.decode(111))
            # print("decode 109:", tokenizer.decode(109))
            # print("xz: ", xz[0])
            # print("xzy: ", xzy[0])
            print("decode x: ", tokenizer.decode(x[0]))
            print("decode xz: ", tokenizer.decode(xz[0]))
            print("decode xzy: ", tokenizer.decode(xzy[0]))
            self.base_model.eval()
            outputs = self.base_model(xzy, labels=xzy_labels, attention_mask=xzy_mask)
            ce_loss, logits = outputs[:2]
            outputs = self.base_model(x, labels=x_labels, attention_mask=x_mask)
            ce_loss_x, logits_x = outputs[:2]
            reward = ce_loss.item() - ce_loss_x.item()
        loss = -reward * (model(xz, labels=xz_labels, attention_mask=xz_mask)[0] - model(x, labels=x_labels, attention_mask=x_mask)[0])
        return loss

    def _prepare_deepspeed(self, model):
        # Adapted from accelerate: https://github.com/huggingface/accelerate/blob/739b135f8367becb67ffaada12fe76e3aa60fefd/src/accelerate/accelerator.py#L1473
        deepspeed_plugin = self.accelerator.state.deepspeed_plugin
        config_kwargs = deepcopy(deepspeed_plugin.deepspeed_config)

        if model is not None:
            if hasattr(model, "config"):
                hidden_size = (
                    max(model.config.hidden_sizes)
                    if getattr(model.config, "hidden_sizes", None)
                    else getattr(model.config, "hidden_size", None)
                )
                if hidden_size is not None and config_kwargs["zero_optimization"]["stage"] == 3:
                    # Note that `stage3_prefetch_bucket_size` can produce DeepSpeed messages like: `Invalidate trace cache @ step 0: expected module 1, but got module 0`
                    # This is expected and is not an error, see: https://github.com/microsoft/DeepSpeed/discussions/4081
                    config_kwargs.update(
                        {
                            "zero_optimization.reduce_bucket_size": hidden_size * hidden_size,
                            "zero_optimization.stage3_param_persistence_threshold": 10 * hidden_size,
                            "zero_optimization.stage3_prefetch_bucket_size": 0.9 * hidden_size * hidden_size,
                        }
                    )

        # If ZeRO-3 is used, we shard both the active and reference model.
        # Otherwise, we assume the reference model fits in memory and is initialized on each device with ZeRO disabled (stage 0)
        if config_kwargs["zero_optimization"]["stage"] != 3:
            config_kwargs["zero_optimization"]["stage"] = 0
        model, *_ = deepspeed.initialize(model=model, config=config_kwargs)
        model.eval()
        return model


@dataclass
class MyDataCollatorWithPadding:
    tokenizer: AutoTokenizer
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        input_ids_q_ls = []
        attention_mask_q_ls = []
        input_ids_a_ls = []
        attention_mask_a_ls = []

        for feature in features:
            input_ids_q_ls.append(feature["input_ids_q"])
            attention_mask_q_ls.append(feature["attention_mask_q"])
            input_ids_a_ls.append(feature["input_ids_a"])
            attention_mask_a_ls.append(feature["attention_mask_a"])
        
        batch = {
            "input_ids_q_l": padding_func(input_ids_q_ls, "left", self.tokenizer.pad_token_id, self.return_tensors),
            "attention_mask_q_l": padding_func(attention_mask_q_ls, "left", 0, self.return_tensors),
            "input_ids_q_r": padding_func(input_ids_q_ls, "right", self.tokenizer.pad_token_id, self.return_tensors),
            "attention_mask_q_r": padding_func(attention_mask_q_ls, "right", 0, self.return_tensors),
            "input_ids_a_l": padding_func(input_ids_a_ls, "left", self.tokenizer.pad_token_id, self.return_tensors),
            "attention_mask_a_l": padding_func(attention_mask_a_ls, "left", 0, self.return_tensors),
            "input_ids_a_r": padding_func(input_ids_a_ls, "right", self.tokenizer.pad_token_id, self.return_tensors),
            "attention_mask_a_r": padding_func(attention_mask_a_ls, "right", 0, self.return_tensors),
        }
        return batch

from transformers import DataCollatorWithPadding
trainer = QTrainer(
    model=model,
    base_model=our_base_model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=MyDataCollatorWithPadding(tokenizer=tokenizer, padding=True)#, max_length=script_args.max_length)
)

trainer.train()

print("Saving last checkpoint of the model")
model.save_pretrained(output_name + "/last_checkpoint2")
trainer.save_model(output_name + "/debug2")
tokenizer.save_pretrained(output_name + "/debug2")
