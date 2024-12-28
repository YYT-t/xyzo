from datasets import load_dataset
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, get_scheduler
from peft import LoraConfig, TaskType, get_peft_model
import multiprocessing
import json, os, re
from datasets import Dataset
import task_configs
from task_configs import task_config_check, task_data_set
import argparse
import torch
from xiaojun_E_step_ent_PPO import MyDataCollatorWithPadding
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="Run multi-GPU inference on questions and answers.")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="The path or identifier of the model to be used for inference.",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="The path of output dataset.",
    )
    parser.add_argument(
        "--sft_data_type",
        type=str,
        required=True,
        help="zgt_raw / zq_raw / zgt_filter / zq_filter",
    )
    parser.add_argument(
        "--task_type",
        type=str,
        default="math_gsm",
        help= "math or code",
    )
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--train_step", type=int, default=999999)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--save_prefix", type=str, required=True)
    parser.add_argument("--with_template", action='store_true')
    return parser.parse_args()


def main(args):
    DEVICE = "cuda:0"
    print ("Loading %s"%args.dataset_path)
    train_dataset = load_dataset(args.dataset_path, split="train")
    print ("Loaded")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path) #AutoTokenizer

    column_names = list(train_dataset.features)
    task_config = task_config_check(args.task_type)

    if args.with_template:
        assert 'Gemma' in str(tokenizer), str(tokenizer)
        def add_template(sample):
            message = [{"role":"user", "content":sample['question']}]
            new_question = tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
            sample['question'] = new_question
            return sample
        train_dataset = train_dataset.map(add_template, num_proc=16)



    if args.sft_data_type == 'zgt_raw':
        def zgt_raw_filter(sample):
            tokenized_q = tokenizer(sample['question'], truncation=True)
            split_ans = sample['answer'].split('\n#### ')
            assert len(split_ans) == 2, sample['answer']
            new_ans = split_ans[0]+'\nThe answer is %s.'%split_ans[1]
            tokenized_a = tokenizer(new_ans, truncation=True)
            sample["input_ids_q"] = tokenized_q["input_ids"]
            sample["attention_mask_q"] = tokenized_q["attention_mask"]
            sample["input_ids_a"] = tokenized_a["input_ids"][1:]
            sample["attention_mask_a"] = tokenized_a["attention_mask"][1:]
            return sample
        train_dataset = train_dataset.map(zgt_raw_filter, remove_columns=column_names, num_proc=16)
    elif args.sft_data_type == 'zq_raw':
        def zq_raw_filter(sample):
            tokenized_q = tokenizer(sample['question'], truncation=True)
            tokenized_a = tokenizer(sample['rational_answer'], truncation=True)
            sample["input_ids_q"] = tokenized_q["input_ids"]
            sample["attention_mask_q"] = tokenized_q["attention_mask"]
            sample["input_ids_a"] = tokenized_a["input_ids"][1:]
            sample["attention_mask_a"] = tokenized_a["attention_mask"][1:]
            return sample
        train_dataset = train_dataset.map(zq_raw_filter, remove_columns=column_names, num_proc=16)
    elif args.sft_data_type == 'zgt_filter':
        def zfilter(sample):
            split_ans = sample['answer'].split('\n#### ')
            assert len(split_ans) == 2, sample['answer']
            gt_str = split_ans[1]
            split_q = sample['rational_answer'].split('The answer is')
            assert len(split_q) >= 2, sample['rational_answer']
            q_reasoning = ' '.join(split_q[:-1])
            return gt_str in q_reasoning
        print ("Before:", len(train_dataset))
        train_dataset = train_dataset.filter(zfilter)
        print ("After:", len(train_dataset))
        def zgt_raw_filter(sample):
            tokenized_q = tokenizer(sample['question'], truncation=True)
            split_ans = sample['answer'].split('\n#### ')
            assert len(split_ans) == 2, sample['answer']
            new_ans = split_ans[0]+'\nThe answer is %s.'%split_ans[1]
            tokenized_a = tokenizer(new_ans, truncation=True)
            sample["input_ids_q"] = tokenized_q["input_ids"]
            sample["attention_mask_q"] = tokenized_q["attention_mask"]
            sample["input_ids_a"] = tokenized_a["input_ids"][1:]
            sample["attention_mask_a"] = tokenized_a["attention_mask"][1:]
            return sample
        train_dataset = train_dataset.map(zgt_raw_filter, remove_columns=column_names, num_proc=16)
    elif args.sft_data_type == 'zq_filter':
        def zfilter(sample):
            split_ans = sample['answer'].split('\n#### ')
            assert len(split_ans) == 2, sample['answer']
            gt_str = split_ans[1]
            split_q = sample['rational_answer'].split('The answer is')
            assert len(split_q) >= 2, sample['rational_answer']
            q_reasoning = ' '.join(split_q[:-1])
            return gt_str in q_reasoning
        print ("Before:", len(train_dataset))
        train_dataset = train_dataset.filter(zfilter)
        print ("After:", len(train_dataset))
        def zq_raw_filter(sample):
            tokenized_q = tokenizer(sample['question'], truncation=True)
            tokenized_a = tokenizer(sample['rational_answer'], truncation=True)
            sample["input_ids_q"] = tokenized_q["input_ids"]
            sample["attention_mask_q"] = tokenized_q["attention_mask"]
            sample["input_ids_a"] = tokenized_a["input_ids"][1:]
            sample["attention_mask_a"] = tokenized_a["attention_mask"][1:]
            return sample
        train_dataset = train_dataset.map(zq_raw_filter, remove_columns=column_names, num_proc=16)
    else:
        raise NotImplementedError()
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=MyDataCollatorWithPadding(tokenizer=tokenizer, padding=True))
    train_steps = min(args.train_step, len(train_dataloader))

    model_config = AutoConfig.from_pretrained(args.model_path)
    print (model_config)
    for key in ('dropout', 'attention_dropout', 'hidden_dropout', 'activation_dropout'):
        if hasattr(model_config, key):
            setattr(model_config, key, 0.0)
    model = AutoModelForCausalLM.from_pretrained(args.model_path, config=model_config, torch_dtype=torch.bfloat16, use_flash_attention_2=False).to(DEVICE).train()
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
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0, betas=(0.9, 0.95))
    scheduler = get_scheduler(name='cosine', optimizer=optimizer, num_warmup_steps=min(100,0.1*train_steps), num_training_steps=train_steps)


    model.train()
    for step, batch in tqdm(enumerate(train_dataloader), total=train_steps):
        if step >= train_steps:
            break

        prompt_input_ids = batch["input_ids_q_l"].to(DEVICE)
        prompt_attention_mask = batch["attention_mask_q_l"].to(DEVICE)
        prompt_length = prompt_input_ids.shape[1]
        answer_input_ids = batch["input_ids_a_r"].to(DEVICE)
        answer_attention_mask = batch["attention_mask_a_r"].to(DEVICE)

        input_ids = torch.cat([prompt_input_ids, answer_input_ids],1)
        attention_mask = torch.cat([prompt_attention_mask, answer_attention_mask],1)
        labels = input_ids.clone()
        labels[:,:prompt_length] = -100
        #print ("============")
        #print ("============")
        #print (tokenizer.decode(input_ids[0][attention_mask[0]!=0]))
        #print (tokenizer.decode(input_ids[0][labels[0]!=-100]))

        output = model.forward(input_ids, attention_mask, labels=labels)
        loss = output.loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        if step % 50 == 0:
            print ("Step: %d, loss: %.4f"%(step,loss.item()))
    model.eval()

    save_path = args.save_prefix + '_' + args.sft_data_type
    if args.with_template:
        save_path = save_path + '_withtemplate'
    model = model.merge_and_unload()
    model.save_pretrained(save_path, from_pt=True)
    tokenizer.save_pretrained(save_path)

if __name__ == '__main__':
    args = parse_args()
    main(args)
