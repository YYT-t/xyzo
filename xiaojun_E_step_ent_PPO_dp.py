import os

import sys
import subprocess
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from task_configs import task_config_check, task_data_set

import torch
import torch.nn as nn
import numpy as np
from copy import deepcopy
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoModel,
    AutoConfig,
    AutoTokenizer,
    HfArgumentParser,
    get_scheduler,
)
from transformers.tokenization_utils_base import PaddingStrategy
from models import AC_Model
# For LoRA
from peft import LoraConfig, TaskType, get_peft_model

# Accelerate
from accelerate import Accelerator
from accelerate.utils import DeepSpeedPlugin

# Your own imports (adjust as needed)
# from task_configs import task_config_check, task_data_set
# from utils import regularized_logp

###############################################################################
# Example functions from your script
###############################################################################

def gather_log_probs(logits, labels):
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    log_probs_labels = log_probs.gather(dim=-1, index=labels.unsqueeze(-1))
    return log_probs_labels.squeeze(-1)

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

def calc_reward_with_nll(ref_model, tokenizer, xz_leftpad, y_rightpad):
    loss_fn = torch.nn.CrossEntropyLoss()
    assert len(xz_leftpad) == len(y_rightpad)
    reward = []
    for one_xz, one_y in zip(xz_leftpad, y_rightpad):
        xz_txt = tokenizer.decode(one_xz, skip_special_tokens=True)
        y_txt = tokenizer.decode(one_y, skip_special_tokens=True)
        concat_toks = tokenizer(xz_txt + y_txt, return_tensors='pt')

        xz_length = tokenizer(xz_txt, return_tensors='pt')['input_ids'].shape[1]
        prompt_length = xz_length + 4  # approximate location: "The answer is"
        output = ref_model.forward(
            concat_toks['input_ids'].to(ref_model.device), 
            attention_mask=concat_toks['attention_mask'].to(ref_model.device)
        )
#        print("output.logits.shape= ", y_txt)
#        print("prompt_length= ", prompt_length)
#        print("output.logits[:,prompt_length-1:-1]= ", output.logits[:,prompt_length-1:-1])
        try:
            shift_logits = output.logits[:,prompt_length-1:-1].view(concat_toks['input_ids'].shape[1]-prompt_length,-1)
            #print(output.logits.size(), concat_toks['input_ids'].size(), prompt_length)
            shift_labels = concat_toks['input_ids'][:,prompt_length:].view(concat_toks['input_ids'].shape[1]-prompt_length).to(ref_model.device)
            loss = loss_fn(shift_logits, shift_labels)
            reward.append(-loss)
        except Exception as e:
            print("error: ", e)
            print(output.logits.size(), concat_toks['input_ids'].size(), prompt_length, xz_txt, "y_txt:", y_txt)
       # shift_labels = concat_toks['input_ids'][:,prompt_length:].view(concat_toks['input_ids'].shape[1]-prompt_length).to(ref_model.device)
       # loss = loss_fn(shift_logits, shift_labels)
       # reward.append(-loss)
    return torch.stack(reward)

class CriticModel(torch.nn.Module):
    def __init__(self, base_model):
        super(CriticModel, self).__init__()
        self.v_head = nn.Linear(base_model.config.hidden_size, 1, bias=False)
        self.v_head = self.v_head.to(torch.bfloat16)
        self.rwtransformer = base_model

    def forward(self, input_ids, attention_mask, return_value_only=False, prompt_length=0):
        outputs = self.rwtransformer(input_ids, attention_mask, use_cache=False)
        values = self.v_head(outputs[0]).squeeze(-1)
        return values

    @property
    def device(self):
        return self.rwtransformer.device

def actor_loss_fn(logprobs, old_logprobs, advantages, mask, cliprange=0.2):
    # policy gradient (PPO clip) loss
    log_ratio = (logprobs - old_logprobs) * mask
    ratio = torch.exp(log_ratio)
    pg_loss1 = -advantages * ratio
    pg_loss2 = -advantages * torch.clamp(ratio, 1.0 - cliprange, 1.0 + cliprange)
    pg_loss = torch.sum(torch.max(pg_loss1, pg_loss2) * mask) / mask.sum()
    return pg_loss

def critic_loss_fn(values, old_values, returns, mask):
    cliprange_value = 0.2
    values_clipped = torch.clamp(
        values, 
        old_values - cliprange_value,
        old_values + cliprange_value,
    )
    vf_loss1 = (values - returns)**2
    vf_loss2 = (values_clipped - returns)**2
    vf_loss = 0.5 * torch.sum(torch.max(vf_loss1, vf_loss2) * mask) / mask.sum()
    return vf_loss

def calc_PPO_loss(model, critic_model, seq, attention_mask, prompts, reward_score, logprobs, ref_logprobs, kl_ctl=0.1):
    # Standard PPO advantage calculation
    rew_clip_val = 5.0
    gamma = 1.0
    lam = 0.95

    with torch.no_grad():
        critic_model.eval()
        old_values = critic_model.forward(
            seq.to(critic_model.device), 
            attention_mask.to(critic_model.device)
        ).detach()[:, :-1]
        start = prompts.size()[-1] - 1
        action_mask = attention_mask[:, 1:]
        ends = start + action_mask[:, start:].sum(1) + 1

        old_rewards = -kl_ctl * (logprobs - ref_logprobs)  # KL reg
        reward_clip = torch.clamp(reward_score, -rew_clip_val, rew_clip_val)
        for i in range(len(old_rewards)):
            old_rewards[i, start:ends[i]][-1] += reward_clip[i]
            old_rewards[i, ends[i]:] = 0
            old_values[i, ends[i]:] = 0

        lastgaelam = 0
        advantages_reversed = []
        length = old_rewards.size()[-1]
        for t in reversed(range(start, length)):
            if t < length - 1:
                nextvalues = old_values[:, t+1]
            else:
                nextvalues = 0.0
            delta = old_rewards[:, t] + gamma * nextvalues - old_values[:, t]
            lastgaelam = delta + gamma * lam * lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)
        returns = advantages + old_values[:, start:]
        advantages = advantages.detach()

    actor_prob = model(input_ids=seq, attention_mask=attention_mask, use_cache=False).logits
    actor_log_prob = gather_log_probs(actor_prob[:, :-1, :], seq[:, 1:])
    actor_loss = actor_loss_fn(
        actor_log_prob[:, start:], 
        logprobs[:, start:], 
        advantages, 
        action_mask[:, start:]
    )

    critic_model.train()
    value = critic_model.forward(
        seq.to(critic_model.device), 
        attention_mask.to(critic_model.device)
    )[:, :-1]
    critic_loss = critic_loss_fn(
        value[:, start:], 
        old_values[:, start:].to(critic_model.device), 
        returns.to(critic_model.device), 
        action_mask[:, start:].to(critic_model.device)
    )
    return actor_loss, critic_loss

###############################################################################
# ScriptArguments
###############################################################################
@dataclass
class ScriptArguments:
    local_rank: Optional[int] = field(
        default=-1, metadata={"help": "Used for multi-gpu"})
    deepspeed: Optional[str] = field(
        default="deepspeed_configs/deepspeed_3.json",
        metadata={"help": "Path to your DeepSpeed JSON config."},
    )
    per_device_train_batch_size: Optional[int] = field(default=2)
    per_device_eval_batch_size: Optional[int] = field(default=1)
    gradient_accumulation_steps: Optional[int] = field(default=4)
    learning_rate: Optional[float] = field(default=5e-7)
    critic_lr: Optional[float] = field(default=5e-7)
    weight_decay: Optional[float] = field(default=0.001)
    model_name: Optional[str] = field(
        default="google/gemma-2b-it",
        metadata={"help": "The model name from HF hub."},
    )
    critic_model_name: Optional[str] = field(default="google/gemma-2-2b-it")
    bf16: Optional[bool] = field(
        default=True,
        metadata={"help": "Whether to use bf16 mixed precision if supported."},
    )
    num_train_epochs: Optional[int] = field(
        default=1,
        metadata={"help": "Number of training epochs."},
    )
    output_suffix: Optional[str] = field(
        default="",
        metadata={"help": "Suffix for the output model directory."},
    )
    gradient_checkpointing: Optional[bool] = field(
        default=True,
        metadata={"help": "Use gradient checkpointing."},
    )
    optim: Optional[str] = field(
        default="adamw_torch",
        metadata={"help": "Which optimizer to use."},
    )
    lr_scheduler_type: Optional[str] = field(
        default="cosine",
        metadata={"help": "Learning rate scheduler."},
    )
    max_length: Optional[int] = field(default=256)
    model_max_length: Optional[int] = field(default=256)
    save_every_steps: Optional[int] = field(default=999999)
    eval_every_steps: Optional[int] = field(default=999999)
    prompt_path: Optional[str] = field(
        default="prompts/math_prompt.txt")
    task_type: Optional[str] = field(
        default="math_metamath")
    ent_coeff: Optional[float] = field(default=0.05)
    temperature: Optional[float] = field(default=0.8)
    num_beams: Optional[int] = field(default=5)
    do_sample: Optional[bool] = field(default=True)
    use_template: Optional[bool] = field(default=False)
    use_lora: Optional[bool] = field(default=True)
    label_smoothing: Optional[float] = field(default=0.0)
    model_path: Optional[str] = field(default="None")
    save_strategy: Optional[str] = field(default="steps")
    wandb_project: Optional[str] = field(default="E_step_ent")
    upload_to_hub: Optional[bool] = field(default=True)

###############################################################################
# Main
###############################################################################
def main(script_args):
    
    # 1) Initialize Accelerator with DeepSpeed config
    ds_plugin = DeepSpeedPlugin(
        hf_ds_config=script_args.deepspeed
    )
    accelerator = Accelerator(
        gradient_accumulation_steps=script_args.gradient_accumulation_steps,
        mixed_precision="bf16" if script_args.bf16 else "no",
        deepspeed_plugin=ds_plugin,
    )

    # 2) Prepare data & tokenizer
    #
    #   Replace these with your own logic (task_config, etc.)
    #
    task_config = task_config_check(script_args.task_type)
    train_set_path, train_dataset = task_data_set(script_args.task_type)
    #train_dataset = train_dataset.map(task_config.tokenize_E(tokenizer), num_proc=16)
    tokenizer = AutoTokenizer.from_pretrained(script_args.model_name)
    tokenizer.model_max_length = script_args.model_max_length
    tokenizer.truncation_side = "left"
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    train_dataset = train_dataset.map(task_config.tokenize_E(tokenizer), num_proc=16)
    # Just a dummy dataset for illustration
    dummy_data = [
        {"input_ids_q":[1,2,3],"attention_mask_q":[1,1,1],"input_ids_a":[4,5],"attention_mask_a":[1,1]},
        {"input_ids_q":[1,2],"attention_mask_q":[1,1],"input_ids_a":[4],"attention_mask_a":[1]}
    ]
    # We treat this as a PyTorch dataset
    from torch.utils.data import Dataset
    """
    class DummyDataset(Dataset):
        def __init__(self, data):
            self.data = data
        def __len__(self):
            return len(self.data)
        def __getitem__(self, idx):
            return {
                "input_ids_q": self.data[idx]["input_ids_q"],
                "attention_mask_q": self.data[idx]["attention_mask_q"],
                "input_ids_a": self.data[idx]["input_ids_a"],
                "attention_mask_a": self.data[idx]["attention_mask_a"]
            }

#    train_dataset = DummyDataset(dummy_data)
    """
    print(train_dataset[0])
    train_steps = len(train_dataset) // (
        script_args.per_device_train_batch_size * script_args.gradient_accumulation_steps
    )

    # 3) Create models
    model_config = AutoConfig.from_pretrained(script_args.model_name)
    for key in ('dropout', 'attention_dropout', 'hidden_dropout', 'activation_dropout'):
        if hasattr(model_config, key):
            setattr(model_config, key, 0.0)

    # Actor model
    model = AutoModelForCausalLM.from_pretrained(
        script_args.model_name,
        config=model_config,
        torch_dtype=torch.bfloat16 if script_args.bf16 else torch.float32,
        use_flash_attention_2=False,
    )
    # Critic model
    critic_base_model = AutoModel.from_pretrained(
        script_args.critic_model_name,
        torch_dtype=torch.bfloat16 if script_args.bf16 else torch.float32,
        use_flash_attention_2=False
    )
    critic_model = CriticModel(critic_base_model)

     # Reference model (frozen)
    ref_model = AutoModelForCausalLM.from_pretrained(
        script_args.model_name,
        config=model_config,
        torch_dtype=torch.bfloat16 if script_args.bf16 else torch.float32,
        use_flash_attention_2=False
    )
    for param in ref_model.parameters():
        param.requires_grad = False
   # ac = AC_Model(model.config, model, critic_model, ref_model)
    ac = AC_Model(AutoConfig.from_pretrained(script_args.model_name), model, critic_model, ref_model)
    ac.actor.train()
    if script_args.use_lora:
        lora_config = LoraConfig(
            r=32,
            lora_alpha=128,
            target_modules=["q_proj", "v_proj"],
            task_type=TaskType.CAUSAL_LM,
            lora_dropout=0.0,
            bias="none",
        )
        ac.actor.enable_input_require_grads()
        ac.actor = get_peft_model(ac.actor, lora_config)

    ac.actor.config.use_cache = not script_args.gradient_checkpointing

    ac.critic.eval()
    ac.ref.eval()

    # 4) Optimizers & Schedulers
    optimizer = torch.optim.AdamW(
        ac.actor.parameters(), 
        lr=script_args.learning_rate, 
        weight_decay=0,
        betas=(0.9, 0.95),
    )
    scheduler = get_scheduler(
        name='cosine',
        optimizer=optimizer,
        num_warmup_steps=min(100,int(0.1*train_steps)),
        num_training_steps=train_steps
    )

    optimizer_critic = torch.optim.AdamW(
        ac.critic.parameters(),
        lr=script_args.critic_lr,
        weight_decay=0,
        betas=(0.9, 0.95),
    )
    scheduler_critic = get_scheduler(
        name='cosine',
        optimizer=optimizer_critic,
        num_warmup_steps=min(100,int(0.1*train_steps)),
        num_training_steps=train_steps
    )

    # 5) Dataloader
    prompt_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=script_args.per_device_train_batch_size,
        collate_fn=MyDataCollatorWithPadding(tokenizer=tokenizer, padding=True),
        shuffle=True,
    )

    # 6) Prepare everything with Accelerator
    (
        ac,
        optimizer,
        optimizer_critic,
        prompt_dataloader,
        scheduler,
        scheduler_critic
    ) = accelerator.prepare(
        ac,
        optimizer,
        optimizer_critic,
        prompt_dataloader,
        scheduler,
        scheduler_critic
    )
    # (Optional) TensorBoard on main process
    if accelerator.is_main_process:
        output_name = script_args.model_path # "./Q_models/deepspeed_test"  # Example
        os.makedirs(output_name, exist_ok=True)
        tb_writer = SummaryWriter(log_dir=os.path.join(output_name, "tb_logs"))
    else:
        tb_writer = None

    # 7) Training Loop (toy example)
    global_step = 0
    for epoch in range(script_args.num_train_epochs):
        for step, batch in tqdm(enumerate(prompt_dataloader), total=len(prompt_dataloader)):
            global_step += 1

            # Generate
            ac.actor.eval()
            with torch.no_grad():
                prompt_input_ids = batch["input_ids_q_l"]
                prompt_attention_mask = batch["attention_mask_q_l"]
                prompt_length = prompt_input_ids.shape[1]
                answer_input_ids = batch["input_ids_a_r"]

                # Generate
                max_min_length = prompt_length + script_args.max_length
                seq = ac.actor.generate(
                    input_ids=prompt_input_ids, 
                    attention_mask=prompt_attention_mask,
                    max_length=max_min_length,
                    pad_token_id=tokenizer.pad_token_id,
                    do_sample=True
                )

                seq_attention_mask = seq.ne(tokenizer.pad_token_id).long()
                seq_attention_mask[:,:prompt_length] = prompt_attention_mask

                # Evaluate reward
                reward_score = calc_reward_with_nll(
                    ac.ref, tokenizer, seq, answer_input_ids
                )

                output = ac.actor(seq, attention_mask=seq_attention_mask)
                output_ref = ac.ref(seq, attention_mask=seq_attention_mask)
                logprobs = gather_log_probs(output.logits[:, :-1, :], seq[:, 1:])
                ref_logprobs = gather_log_probs(output_ref.logits[:, :-1, :], seq[:, 1:])

            # PPO losses
            ac.actor.train()
            actor_loss, critic_loss = calc_PPO_loss(
                ac.actor, ac.critic, seq, seq_attention_mask,
                prompt_input_ids, reward_score, logprobs, ref_logprobs, kl_ctl=0.1
            )

            # Backward
            if global_step >= 50:
                accelerator.backward(actor_loss)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            accelerator.backward(critic_loss)
            optimizer_critic.step()
            scheduler_critic.step()
            optimizer_critic.zero_grad()

            # Logging
            if tb_writer is not None:
                tb_writer.add_scalar('actor_loss', actor_loss.item(), global_step=global_step)
                tb_writer.add_scalar('critic_loss', critic_loss.item(), global_step=global_step)
                tb_writer.add_scalar('reward_mean', reward_score.mean().item(), global_step=global_step)

    # 8) Final save on main process
    if accelerator.is_main_process:
        final_dir = os.path.join(output_name, "final_checkpoint")
        os.makedirs(final_dir, exist_ok=True)
        
        # Unwrap model from Accelerate
        final_ac = accelerator.unwrap_model(ac)
        final_model = final_ac.actor
        # final_model = accelerator.unwrap_model(model)
        if script_args.use_lora:
            final_model = final_model.merge_and_unload()
        final_model.save_pretrained(final_dir)
        tokenizer.save_pretrained(final_dir)
        print("saved to", final_dir)
    if tb_writer is not None:
        tb_writer.close()


if __name__ == '__main__':
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]
    main(script_args)

