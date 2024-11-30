

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from diffuser.datasets.bits_fun.bits_utils import *

from tqdm import tqdm

import os

# Load the fine-tuned GPT-2 model and tokenizer
# model_path = "./saved_model"

import diffuser.utils as utils

class Parser(utils.Parser):
    dataset: str = 'tamp_easy_78'
    config: str = 'config_bits.diffusion.train_diff'

args = Parser().parse_args('diffusion')

log_save_dir = args.savepath + "_transformer/"

# list the folder in log_save_dir
checkpoint_name = os.listdir(log_save_dir)
# then compare the string with the latest checkpoint
checkpoint_name.sort()
model_path = os.path.join(log_save_dir, checkpoint_name[-1])


model = AutoModelForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Load custom_texts from dataset.txt
custom_texts = load_custom_texts(args.train_data_dir)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    model.resize_token_embeddings(len(tokenizer))

def generate_follow_up_steps(prompt, max_length=1000, temperature=0.05, top_p=0.95):
    model.eval()
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    # Check if input_ids is empty
    if input_ids.numel() == 0:
        print(f"Empty input for prompt: {prompt}")
        return prompt  # Return prompt as-is if input_ids is empty

    try:
        # Generate follow-up steps
        outputs = model.generate(
            input_ids,
            max_length=max_length,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.pad_token_id
        )

        # Decode and return the generated steps following the prompt
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text
    except Exception as e:
        print(f"Error during generation for prompt '{prompt}': {e}")
        return prompt
    

def get_prompt(text, first_rows=1):
    splitted =  text.split("\n")[:first_rows]
    # then convert it back to string with \n
    return "\n".join(splitted)

all_texts =""

# Evaluate the model by generating follow-up steps
for i, task in tqdm(enumerate(custom_texts), total=len(custom_texts)):
    # 1. how many rows are for the condition? 0:7
    prompt = get_prompt(task, first_rows=7)
    generated_steps = generate_follow_up_steps(prompt)

    all_texts += f"{generated_steps}\n"

    # print(f"Input Prompt: {prompt}")
    # print("\n")
    # print(f"Generated Steps: {generated_steps}")
    # print("\n")
    # print(f"Ground Truth: {task}")
    # print("----" * 10)

# Save the generated steps to a text file
output_file = os.path.join(log_save_dir, "generated_steps.txt")
with open(output_file, "w") as f:
    f.write(all_texts)

print(f"Generated steps saved to {output_file}")

