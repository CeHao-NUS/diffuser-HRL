from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from diffuser.datasets.bits_fun.bits_utils import *
from tqdm import tqdm
import os
import diffuser.utils as utils
import torch.nn as nn

# Load custom_parser from diffuser.utils
class Parser(utils.Parser):
    dataset: str = 'tamp_easy_78'
    config: str = 'config_bits.diffusion.train_diff'

args = Parser().parse_args('diffusion')

# Automatically get the savepath from the config file
model_save_dir = args.savepath + "_transformer/"
results_save_dir = args.savepath + "_transformer_results/"

# List the folder in model_save_dir
checkpoint_name = os.listdir(model_save_dir)
# Sort the checkpoint names to get the latest one
checkpoint_name.sort()
model_path = os.path.join(model_save_dir, checkpoint_name[-1])

# Load the fine-tuned GPT-2 model and tokenizer
model = AutoModelForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Set padding side to left for decoder-only architecture
tokenizer.padding_side = "left"

# Ensure padding token is set (if not already)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token  # Use eos_token as pad_token

# Set up multi-GPU support using DataParallel
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs!")
    model = nn.DataParallel(model)  # This will distribute the model over available GPUs
else:
    model.to(device)

# Load custom_texts from dataset.txt
custom_texts = load_custom_texts(args.train_data_dir)

# Function to generate follow-up steps for a batch of prompts
def generate_follow_up_steps(prompts, max_length=1000, temperature=0.05, top_p=0.95):
    model.eval()

    # Tokenize all prompts at once with padding and attention mask
    encoding = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
    input_ids = encoding.input_ids.to(device)
    attention_mask = encoding.attention_mask.to(device)  # Get attention mask

    # Check if input_ids is empty
    if input_ids.numel() == 0:
        print(f"Empty input for prompts.")
        return prompts  # Return prompts as-is if input_ids is empty

    try:
        # Generate follow-up steps for the entire batch
        outputs = model.module.generate(  # Access the underlying model when using DataParallel
            input_ids,
            attention_mask=attention_mask,  # Pass the attention mask
            max_length=max_length,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.pad_token_id
        )

        # Decode and return the generated steps for all prompts
        generated_texts = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
        return generated_texts
    except Exception as e:
        print(f"Error during batch generation: {e}")
        return prompts

# Function to extract the first rows from the task text
def get_prompt(text, first_rows=1):
    splitted = text.split("\n")[:first_rows]
    return "\n".join(splitted)

# Ensure the result directory exists
os.makedirs(results_save_dir, exist_ok=True)

# Main evaluation loop to process custom texts
all_texts = ""

# Prepare the batch of prompts
batch_size = 16  # Adjust batch size based on your GPU memory
prompts_batch = []

# Specify the output file path
output_file = os.path.join(results_save_dir, "generated_steps.txt")

for i, task in tqdm(enumerate(custom_texts), total=len(custom_texts)):
    prompt = get_prompt(task, first_rows=7)
    prompts_batch.append(prompt)

    # When the batch is full or it's the last task, generate results
    if len(prompts_batch) == batch_size or i == len(custom_texts) - 1:
        # Send the batch to the device (GPU) before generation
        # prompts_batch_device = [prompt.to(device) for prompt in prompts_batch]

        # Generate steps
        generated_steps = generate_follow_up_steps(prompts_batch)

        # Prepare the output text (joining generated steps with newline separator)
        output_text = "\n".join(generated_steps) + "\n"

        # Append the generated steps to the same text file
        with open(output_file, "a") as f:  # Open in append mode ("a")
            f.write(output_text)

        print(f"Generated steps for batch {i // batch_size + 1} saved to {output_file}")

        # Clear the batch for the next iteration
        prompts_batch = []
