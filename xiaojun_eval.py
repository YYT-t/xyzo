from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from task_configs import task_config_check, task_data_set
import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset
#import deepspeed
from copy import deepcopy
# from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoModel,
    AutoConfig,
    AutoTokenizer,
    HfArgumentParser,
    PreTrainedTokenizerBase,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    GPT2Tokenizer, GPT2LMHeadModel,
    Gemma2ForCausalLM,
    get_scheduler,
)
from transformers.utils import PaddingStrategy
#import wandb
import sys
import os
from utils import regularized_logp
from torch.utils.tensorboard import SummaryWriter
from transformers import DataCollatorWithPadding
import subprocess
#import lm_eval
from tqdm import tqdm

@dataclass
class ScriptArguments:
    """
    These arguments vary depending on how many GPUs you have, what their capacity and features are, and what size model you want to train.
    """
    batch_size: Optional[int] = field(default=4)
    pmodel_name: Optional[str] = field(default="google/gemma-2-2b-it")
    qmodel_name: Optional[str] = field(default="google/gemma-2-2b-it")
    out_path: Optional[str] = field(default="./out.txt")
    task_type: Optional[str] = field(default="math_gsm")
    max_length: Optional[int] = field(default=256)

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

def calc_all_metrics(ref_model, tokenizer, xz_leftpad, y_rightpad):
    loss_fn = torch.nn.CrossEntropyLoss()
    assert len(xz_leftpad) == len(y_rightpad)
    metrics = []

    for one_xz, one_y in zip(xz_leftpad, y_rightpad):
        # TODO: calculating without batch for convenience; can batchify to accelerate
        xz_txt = tokenizer.decode(one_xz, skip_special_tokens=True)
        y_txt = tokenizer.decode(one_y, skip_special_tokens=True)
        concat_toks = tokenizer(xz_txt+y_txt, return_tensors='pt')

        #prompt_length = xz_length
        # Designing mask: only consider the tokens after "The answer is"
        xz_length = tokenizer(xz_txt, return_tensors='pt')['input_ids'].shape[1]
        prompt_length = xz_length + 4
        assert y_txt.startswith('The answer is')
        # TODO: The length is approximate and may not be correct, so the assert below may fail... just ignoring it for now.
        #assert tokenizer.decode(concat_toks['input_ids'][0][xz_length:prompt_length]) == 'The answer is ', [tokenizer.decode(concat_toks['input_ids'][0][xz_length:prompt_length])]
        output = ref_model.forward(concat_toks['input_ids'].to(ref_model.device), attention_mask=concat_toks['attention_mask'].to(ref_model.device))

        shift_logits = output.logits[:,prompt_length-1:-1]
        shift_labels = concat_toks['input_ids'][:,prompt_length:].to(ref_model.device)
        full_nll = loss_fn(shift_logits.view(concat_toks['input_ids'].shape[1]-prompt_length,-1), shift_labels.view(concat_toks['input_ids'].shape[1]-prompt_length))
        next_tok_logp = -loss_fn(shift_logits[:,0], shift_labels[:,0])

        next_tok_pred = shift_logits[:,0].argmax(1)
        next_tok_acc = (next_tok_pred==shift_labels[:,0]).float().mean()

        #metrics.append({'full_nll':full_nll, 'next_tok_logp':next_tok_logp, 'next_tok_acc':next_tok_acc})

        # TODO: prompt_gen_acc
        new_prompt_toks = tokenizer(xz_txt+'The answer is', return_tensors='pt')
        with torch.no_grad():
            # Generate
            seq = ref_model.generate(input_ids=new_prompt_toks['input_ids'].to(ref_model.device), attention_mask=new_prompt_toks['attention_mask'].to(ref_model.device), max_length=new_prompt_toks['input_ids'].shape[1]+20, pad_token_id=tokenizer.pad_token_id, do_sample=False)
            gen_txt = tokenizer.decode(seq[0], skip_special_tokens=True)
            true_ans = y_txt[len('The answer is'):-1] # remove the final period
            gen_ans = gen_txt[len(xz_txt+'The answer is'):][:len(true_ans)]
            prompt_gen_acc = torch.FloatTensor([gen_ans==true_ans])
        metrics.append({'full_nll':full_nll, 'next_tok_logp':next_tok_logp, 'next_tok_acc':next_tok_acc, 'prompt_gen_acc':prompt_gen_acc})

    return metrics


def main(script_args):
    DEVICE = "cuda:0"
    tokenizer_name = script_args.pmodel_name
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name) #AutoTokenizer
    tokenizer.model_max_length = 512
    tokenizer.truncation_side = "left"
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token

    task_config = task_config_check(script_args.task_type)
    train_set_path, train_dataset = task_data_set(script_args.task_type)
    train_dataset = train_dataset.map(task_config.tokenize_E(tokenizer), num_proc=16)
    prompt_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=script_args.batch_size, collate_fn=MyDataCollatorWithPadding(tokenizer=tokenizer, padding=True))

    # assuming that p and q are from the same class
    model_config = AutoConfig.from_pretrained(script_args.pmodel_name)
    VOCAB_SIZE = model_config.vocab_size
    print (model_config)
    for key in ('dropout', 'attention_dropout', 'hidden_dropout', 'activation_dropout'):
        if hasattr(model_config, key):
            setattr(model_config, key, 0.0)
    model = AutoModelForCausalLM.from_pretrained(script_args.qmodel_name, config=model_config, torch_dtype=torch.bfloat16, use_flash_attention_2=False).to(DEVICE).eval()
    ref_model = AutoModelForCausalLM.from_pretrained(script_args.pmodel_name, config=model_config, torch_dtype=torch.bfloat16, use_flash_attention_2=False).to(DEVICE).eval()

    all_metrics = []

    with open(script_args.out_path, 'w') as outf:
        for step, batch in tqdm(enumerate(prompt_dataloader), total=len(prompt_dataloader)):
            prompt_input_ids = batch["input_ids_q_l"].to(DEVICE)
            prompt_attention_mask = batch["attention_mask_q_l"].to(DEVICE)
            prompt_length = prompt_input_ids.shape[1]
            answer_input_ids = batch["input_ids_a_r"].to(DEVICE)

            model.eval()
            max_min_length = prompt_length + script_args.max_length
            with torch.no_grad():
                # Generate
                seq = model.generate(input_ids=prompt_input_ids, attention_mask=prompt_attention_mask, max_length=max_min_length, pad_token_id=tokenizer.pad_token_id, do_sample=True)
                seq_attention_mask = seq.not_equal(tokenizer.pad_token_id).long()
                seq_attention_mask[:,:prompt_length] = prompt_attention_mask
                cur_metrics = calc_all_metrics(ref_model, tokenizer, seq, answer_input_ids)
            all_metrics.extend(cur_metrics)
            
            for one_seq in seq:
                outf.write(tokenizer.decode(one_seq, skip_special_tokens=True)+"\n")
                outf.write("================\n")
                outf.write("================\n")
                outf.write("================\n")

            #if step > 10:
            #    break

        keys = [k for k in all_metrics[0]]
        processed_metrics = {k:[] for k in keys}
        for metric in all_metrics:
            for k in keys:
                processed_metrics[k].append(metric[k])
        metric_results = {}
        for k in keys:
            metric_results[k] = torch.stack(processed_metrics[k]).mean()
        print (metric_results)
        outf.write(str(metric_results))

if __name__ == '__main__':
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]

    main(script_args)
