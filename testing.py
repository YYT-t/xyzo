from vllm import LLM, SamplingParams
from datasets import load_dataset

from transformers import AutoTokenizer
instruct_prompt = r"Answer the question based on the following example:"
example1 = r"""Question: Jack is stranded on a desert island. He wants some salt to season his fish. He collects 2 liters of seawater in an old bucket. If the water is 20% salt, how many ml of salt will Jack get when all the water evaporates? Answer: First find how many liters of the seawater are salt: 2 liters * 20% = 0.4 liters Then multiply that amount by 1000 ml/liter to find the number of ml of salt Jack gets: 0.4 liters * 1000 ml/liter = 400 ml."""
example2 = r"""Question: Samantha’s last name has three fewer letters than Bobbie’s last name. If Bobbie took two letters off her last name, she would have a last name twice the length of Jamie’s. Jamie’s full name is Jamie Grey. How many letters are in Samantha’s last name? Answer: There are 4 letters in Jamie’s last name, so Bobbie’s name is 4*2 +2 = 10 letters long. Samantha’s last name is 3 letters shorter than Bobbie’s, so there are 10 - 3 = 7 letters in Samantha’s last name."""
few_shot_cot_prompt = instruct_prompt + '\n' + example2 + f'\nQuestion: '  #'\n' + example1

def tokenize(sample):
    answer_text = sample['answer'].split('####')[-1].strip()
    sample["few_shot_cot_question"] = few_shot_cot_prompt + sample['question']
    sample["answer_text"] = f"The answer is {answer_text}."
    return sample

train_path = "openai/gsm8k"
dataset_ = load_dataset(train_path, "main")["train"]
dataset_ = dataset_.map(tokenize, num_proc=16)
questions = dataset_["few_shot_cot_question"]
answers = dataset_["answer_text"]

model_name =  "YYT-t/gemma-1.1-7b-it_MetaMathQA_ent0.05_beam1_dosampleFalse_temp0.8_estep__epoch1" #"Q_models/tt_3/debug2"  # "Qwen/Qwen2.5-Math-1.5B"#"facebook/opt-125m"#"google/gemma-2-2b-it"#"deepseek-ai/deepseek-math-7b-rl"
tokenizer = AutoTokenizer.from_pretrained(model_name)
sampling_params = SamplingParams(
    # temperature=0.0,
    # top_p=1.0,
    # top_k=-1,
    # seed=42,
    max_tokens=512,
    min_tokens=100,
    # n=1,
    # frequency_penalty=1.0,
   # stop_token_ids=[tokenizer.eos_token_id],
   stop=['Question:'],
)
# print("questions:", questions)
llm = LLM(model=model_name, tokenizer=model_name, dtype="bfloat16", seed=42, gpu_memory_utilization=0.9)
few_shot_questions = questions[9]
print("few_shot_questions:", few_shot_questions)
print("decode:", [tokenizer.decode(108)])
rational = llm.generate(few_shot_questions, sampling_params, use_tqdm=True)
print("rational:", rational)
print(rational[0].outputs[0].text)