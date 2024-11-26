import abc

class Config_Math(abc):
    def __init__(self):
        self.stop_str_gen_z = ["Question:"]
        self.prompt_path = "prompts/math_prompt.txt"
    def tokenize_E(self):
        pass
    def M_sft_cot_prefix(self):
        def cot_prefix(sample):
            sample["text"] = 'Question: ' + sample["question"] + ' Answer: ' + sample["answer"]
        #    sample["prompt"] = few_shot_cot_prompt + sample["question"]
        #    sample["completion"] = sample["rational_answer"]
            return sample
        return cot_prefix
    def inference_tokenize(self):
        def tokenize(sample):
            answer_text = sample['response'].split("The answer is")[-1].strip()
            sample["few_shot_cot_question"] = self.few_shot_cot_prompt + sample['query']
            sample["answer_text"] = f"The answer is {answer_text}."
            return sample
        return tokenize
class Config_Math_GSM(Config_Math):
    def __init__(self):
        with open(self.prompt_path, "r") as file:
            # Read the content of the file
            self.few_shot_cot_prompt = file.read()



class Config_Math_MetaMath(Config_Math):
    def __init__(self):
        with open(self.prompt_path, "r") as file:
            # Read the content of the file
            self.few_shot_cot_prompt = file.read()


    def tokenize_E(self,tokenizer):
        
        def tokenize(sample):
            tokenized_q = tokenizer(self.few_shot_cot_prompt + sample['query'], truncation=True)
            answer_text = sample['response'].split('The answer is: ')[-1].strip()
            answer = f"The answer is {answer_text}."
            tokenized_a = tokenizer(answer, truncation=True)
            sample["input_ids_q"] = tokenized_q["input_ids"]
            sample["attention_mask_q"] = tokenized_q["attention_mask"]
            sample["input_ids_a"] = tokenized_a["input_ids"]
            sample["attention_mask_a"] = tokenized_a["attention_mask"]
            return sample
        return tokenize

class Config_Code(Config_Math):
    def __init__(self):
        self.stop_str_gen_z = ["""[Implementation]"""]
        self.prompt_path = "prompts/code_prompt.txt"
    def tokenize_E(self):
        pass
    def M_sft_cot_prefix(self):
        pass
    def inference_tokenize(self):
        pass

def task_config_check(task_name):
    if task_name == "math_gsm":
        return Config_Math_GSM()
    elif  task_name == "math_metamath":
        return Config_Math_MetaMath() 
    elif task_name == "code":
        return Config_Code()
    else:
        raise(NotImplementedError)