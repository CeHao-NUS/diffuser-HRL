import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load the model and tokenizer
model_name = "gpt2"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Set the pad_token to be the same as eos_token for GPT-2
tokenizer.pad_token = tokenizer.eos_token

# Wrap the model with DataParallel to use multiple GPUs
model = torch.nn.DataParallel(model)  # This will automatically use all available GPUs (cuda:0, cuda:1, etc.)

# Move the model to the GPU (it will be distributed across all available GPUs)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Create a batch of input texts
input_texts = ["Hello, world!", "How are you?", "What is your name?", "Tell me a joke.", "Give me a quote."] * 1000  # Large batch
inputs = tokenizer(input_texts, padding=True, return_tensors="pt")

# Move input tensors to the same device as the model
input_ids = inputs.input_ids.to(device)
attention_mask = inputs.attention_mask.to(device)

# Use model.generate for text generation (DataParallel will distribute the batch across GPUs)
generated_ids = model.module.generate(input_ids, attention_mask=attention_mask, max_length=50)

# Decode and print generated text
generated_texts = [tokenizer.decode(ids, skip_special_tokens=True) for ids in generated_ids]
for i, generated_text in enumerate(generated_texts):
    print(f"Generated text {i+1}: {generated_text}")
