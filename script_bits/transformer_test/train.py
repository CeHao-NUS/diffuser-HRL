from diffuser.datasets.bits_fun.bits_utils import *
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import Dataset
import torch
import os
from datetime import datetime

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import diffuser.utils as utils

class Parser(utils.Parser):
    dataset: str = 'tamp_easy_78'
    config: str = 'config_bits.diffusion.train_diff'

args = Parser().parse_args('diffusion')

log_save_dir = args.savepath + "_transformer/"

# timestamp = datetime.now().strftime('%m-%d-%H-%M')
# log_save_dir = args.savepath + timestamp

# Step 1: Define function to load custom texts from dataset.txt
custom_texts = load_custom_texts(args.train_data_dir)

# Step 2: Initialize the GPT-2 model and tokenizer
model_name = "gpt2"  # Use GPT-2 model
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  # Use EOS token as padding token

# Step 3: Preprocess the data by tokenizing and setting labels
def tokenize_function(example):
    inputs = tokenizer(example["text"], padding="max_length", truncation=True, max_length=128)
    inputs["labels"] = inputs["input_ids"].copy()  # Set labels for causal language modeling
    return inputs

# Convert custom text data to dataset format and tokenize
dataset = Dataset.from_dict({"text": custom_texts}).map(tokenize_function, remove_columns=["text"])

# Step 4: Split dataset into 80% train and 20% eval
train_dataset = dataset.train_test_split(test_size=0.05)["train"]
eval_dataset = dataset.train_test_split(test_size=0.05)["test"]

# Step 5: Calculate save_steps based on the number of epochs and dataset size
num_epochs = 100
batch_size = 32
num_training_steps = (len(train_dataset) // batch_size) * num_epochs
# save_every_n_epochs = 10
# save_steps = num_training_steps // (num_epochs / save_every_n_epochs)
save_steps = 100

# Step 6: Set up training arguments, including logging strategy and evaluation strategy
training_args = TrainingArguments(
    output_dir=log_save_dir,  # Output directory
    overwrite_output_dir=True,
    num_train_epochs=num_epochs,
    per_device_train_batch_size=batch_size,
    save_steps=save_steps,
    save_total_limit=2,
    logging_dir=log_save_dir,  # Directory to save logs
    logging_steps=10,          # Log training loss every 10 steps
    logging_strategy="steps",  # Log based on steps rather than epochs
    evaluation_strategy="steps",  # Evaluate periodically
    eval_steps=10,            # Evaluate every 500 steps (adjust as needed)
    prediction_loss_only=True, # Only calculate the loss
)

# Step 7: Define a custom Trainer class to compute the loss if needed
class CustomTrainer(Trainer):
    def save_model(self, output_dir=None, _internal_call=False):
        # Call the parent method to save the model
        super().save_model(output_dir, _internal_call=_internal_call)
        
        # Save the tokenizer only when output_dir is provided
        if output_dir is not None:
            tokenizer.save_pretrained(output_dir)

        print(f"Model and tokenizer saved in {output_dir}")

# Step 8: Initialize the Trainer with the train and eval datasets
trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,  # Pass the eval dataset for evaluation
)

# Step 9: Train the model
trainer.train()

# Step 10: Save the trained model and tokenizer
# trainer.save_model(model_save_path)  # Save model to directory "saved_model"
# tokenizer.save_pretrained(model_save_path)  # Save tokenizer
