from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from task_configs import task_config_check, task_data_set
# import evaluate
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
from peft import LoraConfig, TaskType, get_peft_model
from tqdm import tqdm

def gather_log_probs(logits, labels):
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    log_probs_labels = log_probs.gather(dim=-1, index=labels.unsqueeze(-1))
    return log_probs_labels.squeeze(-1)


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
    per_device_train_batch_size: Optional[int] = field(default=1)
    per_device_eval_batch_size: Optional[int] = field(default=1)
    gradient_accumulation_steps: Optional[int] = field(default=8)
    learning_rate: Optional[float] = field(default=5e-7)
    critic_lr: Optional[float] = field(default=5e-7)
    weight_decay: Optional[float] = field(default=0.001)
    model_name: Optional[str] = field(
        default="google/gemma-2b-it",  # "mistralai/Mistral-7B-Instruct-v0.2",
        metadata={
            "help": "The model that you want to train from the Hugging Face hub. E.g. gpt2, gpt2-xl, bert, etc."
        },
    )
    critic_model_name: Optional[str] = field(default="google/gemma-2-2b-it")
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
    use_lora: Optional[bool] = field(default=True)
    label_smoothing: Optional[float] = field(default=0.0)
    model_path: Optional[str] = field(default="None")
    save_strategy: Optional[str] = field(default="steps")
    wandb_project: Optional[str] = field(default="E_step_ent")
    upload_to_hub: Optional[bool] = field(default=True)

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
        # assert tokenizer.decode(concat_toks['input_ids'][0][xz_length:prompt_length]) == 'The answer is', [tokenizer.decode(concat_toks['input_ids'][0][xz_length:prompt_length])]
        output = ref_model.forward(concat_toks['input_ids'].to(ref_model.device), attention_mask=concat_toks['attention_mask'].to(ref_model.device))
        if output.logits.shape[1] <= prompt_length:
            prompt_length = output.logits.shape[1] - 1
        print(concat_toks['input_ids'].size(), y_txt)
        print("concat_toks['input_ids'].shape[1]-prompt_length=", concat_toks['input_ids'].shape[1]-prompt_length)
        print("output.logits.shape[1]=", output.logits.shape[1])
        print("prompt_length=", prompt_length)
        shift_logits = output.logits[:,prompt_length-1:-1].view(concat_toks['input_ids'].shape[1]-prompt_length,-1)
        shift_labels = concat_toks['input_ids'][:,prompt_length:].view(concat_toks['input_ids'].shape[1]-prompt_length).to(ref_model.device)
        loss = loss_fn(shift_logits, shift_labels)
        reward.append(-loss)
    return torch.stack(reward)


class CriticModel(torch.nn.Module):
    def __init__(self, base_model):
        super(CriticModel, self).__init__()
        #self.v_head = nn.Linear(self.config.word_embed_proj_dim,1,bias=False)
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
    ## policy gradient loss
    log_ratio = (logprobs - old_logprobs) * mask
    ratio = torch.exp(log_ratio)
    pg_loss1 = -advantages * ratio
    pg_loss2 = -advantages * torch.clamp(ratio, 1.0 - cliprange,
                                         1.0 + cliprange)
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
    vf_loss = 0.5 * torch.sum(
        torch.max(vf_loss1, vf_loss2) * mask) / mask.sum()
    return vf_loss

def calc_PPO_loss(model, critic_model, seq, attention_mask, prompts, reward_score, logprobs, ref_logprobs, kl_ctl=0.1):
    rew_clip_val = 5.0
    gamma = 1.0
    lam = 0.95

    with torch.no_grad():
        # Calculate advantage
        critic_model.eval()
        old_values = critic_model.forward(seq.to(critic_model.device), attention_mask.to(critic_model.device)).to(model.device).detach()[:,:-1]
        start = prompts.size()[-1] - 1
        action_mask = attention_mask[:,1:]
        ends = start + action_mask[:,start:].sum(1)+1

        old_rewards = -kl_ctl * (logprobs - ref_logprobs)  # KL reg
        reward_clip = torch.clamp(reward_score, -rew_clip_val, rew_clip_val)
        for i in range(len(old_rewards)):
            old_rewards[i,start:ends[i]][-1] += reward_clip[i].to(old_rewards.device)
            old_rewards[i,ends[i]:] = 0
            old_values[i,ends[i]:] = 0

        lastgaelam = 0
        advantages_reversed = []
        length = old_rewards.size()[-1]
        for t in reversed(range(start, length)):
            nextvalues = old_values[:,t+1] if t < length-1 else 0.0
            delta = old_rewards[:,t] + gamma * nextvalues - old_values[:,t]
            lastgaelam = delta + gamma * lam * lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)
        returns = advantages + old_values[:, start:]
        advantages = advantages.detach()

    actor_prob = model(input_ids=seq, attention_mask=attention_mask, use_cache=False).logits
    actor_log_prob = gather_log_probs(actor_prob[:,:-1,:], seq[:,1:])
    actor_loss = actor_loss_fn(actor_log_prob[:,start:], logprobs[:,start:], advantages, action_mask[:,start:])
    critic_model.train()
    value = critic_model.forward(input_ids=seq.to(critic_model.device), attention_mask=attention_mask.to(critic_model.device))[:, :-1]
    critic_loss = critic_loss_fn(value[:,start:], old_values[:,start:].to(critic_model.device), returns.to(critic_model.device), action_mask[:,start:].to(critic_model.device))
    return actor_loss, critic_loss


def main(script_args):
    # get cuda_visible_devices
    cuda_visible_devices = os.getenv("CUDA_VISIBLE_DEVICES", "")
    # print(f"cuda_visible_devices: {cuda_visible_devices}")
    cuda_visible_devices = cuda_visible_devices.split(",")
    Devices = [f"cuda:{i}" for i in cuda_visible_devices]
    print(f"Devices: {Devices}")

    print("script_args.task_type=", script_args.task_type)
    task_config = task_config_check(script_args.task_type)
    train_set_path, train_dataset = task_data_set(script_args.task_type)

    tokenizer_name = script_args.model_name
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name) #AutoTokenizer

    tokenizer.model_max_length = script_args.model_max_length
    tokenizer.truncation_side = "left"
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    if "llama" in script_args.model_name.lower():
        stop_strings = ["<|eot_id|>"]
        stop_tokens = []
        for stop_string in stop_strings:
            stop_tokens += tokenizer(stop_string)["input_ids"]
        print(stop_tokens)


    base_model_name = script_args.model_name.split("/")[1]
    data_name = train_set_path.split("/")[1]
    if script_args.model_path == "None":
        trained_model_name = f"{base_model_name}_{data_name}_ent{script_args.ent_coeff}_\
    beam{script_args.num_beams}_dosample{script_args.do_sample}_temp{script_args.temperature}_labelsm{script_args.label_smoothing}_\
    totalepoch{script_args.num_train_epochs}"
        output_name = f"./Q_models/{trained_model_name}"
    else:
        trained_model_name = script_args.model_path.split("/")[-1]
        output_name = script_args.model_path
        
    if not os.path.exists(output_name):
        os.makedirs(output_name)

    train_dataset = train_dataset.map(task_config.tokenize_E(tokenizer), num_proc=16)
    train_steps = len(train_dataset) // (script_args.per_device_train_batch_size * script_args.gradient_accumulation_steps)

    model_config = AutoConfig.from_pretrained(script_args.model_name)
    VOCAB_SIZE = model_config.vocab_size
    print (model_config)
    for key in ('dropout', 'attention_dropout', 'hidden_dropout', 'activation_dropout'):
        if hasattr(model_config, key):
            setattr(model_config, key, 0.0)
    
    model = AutoModelForCausalLM.from_pretrained(script_args.model_name, config=model_config, torch_dtype=torch.bfloat16, use_flash_attention_2=False).to(Devices[0]).train()
    if script_args.use_lora:
        lora_config = LoraConfig(
            r=32,
            lora_alpha=128,
            target_modules=["q_proj", "v_proj"],
            task_type=TaskType.CAUSAL_LM,
            lora_dropout=0.0,
            bias="none",
        )
        model.enable_input_require_grads()
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    model.config.use_cache = not script_args.gradient_checkpointing
    optimizer = torch.optim.AdamW(model.parameters(), lr=script_args.learning_rate, weight_decay=0, betas=(0.9, 0.95))
    scheduler = get_scheduler(name='cosine', optimizer=optimizer, num_warmup_steps=min(100,0.1*train_steps), num_training_steps=train_steps)

    ref_model = AutoModelForCausalLM.from_pretrained(script_args.model_name, config=model_config, torch_dtype=torch.bfloat16, use_flash_attention_2=False).to(Devices[1]).eval()
    ref_model.requires_grad = False

    critic_base_model = AutoModel.from_pretrained(script_args.critic_model_name, torch_dtype=torch.bfloat16, use_flash_attention_2=False).to(Devices[2])
    critic_model = CriticModel(critic_base_model).to(Devices[3]).eval()
    optimizer_critic = torch.optim.AdamW(critic_model.parameters(), lr=script_args.critic_lr, weight_decay=0, betas=(0.9, 0.95))
    scheduler_critic = get_scheduler(name='cosine', optimizer=optimizer_critic, num_warmup_steps=min(100,0.1*train_steps), num_training_steps=train_steps)


    prompt_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=script_args.per_device_train_batch_size, collate_fn=MyDataCollatorWithPadding(tokenizer=tokenizer, padding=True))
    tb_writer = SummaryWriter(log_dir='%s/tb_log/'%script_args.model_path)
    for step, batch in tqdm(enumerate(prompt_dataloader), total=len(prompt_dataloader)):
        prompt_input_ids = batch["input_ids_q_l"]
        prompt_attention_mask = batch["attention_mask_q_l"]
        prompt_length = prompt_input_ids.shape[1]
        answer_input_ids = batch["input_ids_a_r"]

        model.eval()
        max_min_length = prompt_length + script_args.max_length
        with torch.no_grad():
            # Generate
            seq = model.generate(input_ids=prompt_input_ids.to(model.device), 
                                 attention_mask=prompt_attention_mask.to(model.device), 
                                 max_length=max_min_length, 
                                 pad_token_id=tokenizer.pad_token_id, 
                                 do_sample=True,
                                 stop_strings=stop_strings,
                                 tokenizer=tokenizer,
                                 )
            for idx in range(len(seq[0])-1, -1, -1):
                if not seq[0][idx] in stop_tokens:
                    seq = seq[:,:idx+1]
                    break
            # print("seq=", seq)
            # print(tokenizer.decode(seq[0]))
            seq_attention_mask = seq.not_equal(tokenizer.pad_token_id).long()
            seq_attention_mask[:,:prompt_length] = prompt_attention_mask

            # Evaluate reward and other info
            reward_score = calc_reward_with_nll(ref_model, tokenizer, seq.to(ref_model.device), answer_input_ids.to(ref_model.device))
            print (reward_score)
            # TODO: hard-coded reward normalization here. May try other methods.
            #reward_score = (reward_score+4.5)*2
            #reward_score = reward_score+4.0
            reward_score = reward_score+0.0
            # add some length encouring factor to reward
            len_reward = []
            for i, one_seq in enumerate(seq):
                cur_len = one_seq.not_equal(tokenizer.pad_token_id).nonzero().max() - prompt_length
                print("cur_len.device: ", cur_len.device)
                print("reward_score[i].device: ", reward_score[i].device)
                reward_score[i] = reward_score[i] + 0.05*min(cur_len.to(reward_score[i].device),40)


            print ("Using a hard-coded simple normalization")
            output = model(seq, attention_mask=seq_attention_mask)
            output_ref = ref_model(seq.to(ref_model.device), attention_mask=seq_attention_mask.to(ref_model.device))
            logprobs = gather_log_probs(output.logits[:,:-1,:],seq[:,1:])
            ref_logprobs = gather_log_probs(output_ref.logits[:,:-1,:].to(seq.device),seq[:,1:])

        # Calculate actor loss and critic loss with standard PPO
        model.train()
        actor_loss, critic_loss = calc_PPO_loss(model, critic_model, seq, seq_attention_mask, prompt_input_ids, reward_score, logprobs, ref_logprobs, kl_ctl=0.1)
        #print (actor_loss)
        #print (critic_loss)

        # TODO: calc entropy reg if needed

        # add KL div to track the current model
        with torch.no_grad():
            prob_p = torch.nn.functional.softmax(output.logits, -1)
            prob_q = torch.nn.functional.softmax(output_ref.logits, -1).to(prob_p.device)
            kl_position_loss = -prob_p * torch.log(prob_q+1e-6)
            position_weight = torch.zeros_like(kl_position_loss)
            position_weight[:,prompt_length:] = 1
            position_weight[seq_attention_mask==0] == 0
            position_weight = position_weight / (position_weight.sum(dim=1,keepdim=True)+1e-8)
            kl_loss = (position_weight*kl_position_loss).sum()

        #print ("TESTING: no actor training!")
        if step >= 50:
            # in the first 50 steps, let critic model learn first
            optimizer.zero_grad()
            actor_loss.backward()
            optimizer.step()
            scheduler.step()
        optimizer_critic.zero_grad()
        critic_loss.backward()
        optimizer_critic.step()
        scheduler_critic.step()
        print ("Step %d, Reward score: %s"%(step, reward_score))
        if step % 10 == 0:
            print ("====================")
            print (tokenizer.decode(prompt_input_ids[0]))
            print ("--------------------")
            print (tokenizer.decode(seq[0]))
            print ("--------------------")
            print (tokenizer.decode(answer_input_ids[0]))

        tb_writer.add_scalar('actor_loss', actor_loss.item(), global_step=step)
        tb_writer.add_scalar('critic_loss', critic_loss.item(), global_step=step)
        tb_writer.add_scalar('kl_loss', kl_loss.item(), global_step=step)
        tb_writer.add_scalar('reward', reward_score.mean().item(), global_step=step)

    final_dir = output_name + "/final_checkpoint"
    print("Saving last checkpoint of the model")
    if script_args.use_lora:
        model = model.merge_and_unload()
    model.save_pretrained(final_dir, from_pt=True)
    tokenizer.save_pretrained(final_dir)
    #trainer.save_model(final_dir)
    subprocess.run([
        "huggingface-cli", "upload", 
        f"YYT-t/{trained_model_name}_final_checkpoint", 
        f"{output_name}/final_checkpoint", 
        "--token", "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"
    ])

if __name__ == '__main__':
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]

    main(script_args)