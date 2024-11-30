import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load the model and tokenizer
model_name = "gpt2"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Set the pad_token to be the same as eos_token for GPT-2
tokenizer.pad_token = tokenizer.eos_token

# Define the custom ManualDataParallel class
class ManualDataParallel:
    def __init__(self, model, device_ids):
        self.model = model
        self.device_ids = device_ids

        # Move model to the first device and replicate it across devices
        self.model = self.model.to(device_ids[0])  # Move model to the first GPU
        self.devices = [torch.device(f'cuda:{device_id}') for device_id in device_ids]

    def forward(self, input_ids, attention_mask):
        num_gpus = len(self.device_ids)
        batch_size = input_ids.size(0)

        # Split the data across GPUs
        input_ids_split = list(torch.chunk(input_ids, num_gpus, dim=0))
        attention_mask_split = list(torch.chunk(attention_mask, num_gpus, dim=0))

        outputs = []

        # Forward pass on each device
        for i, dev in enumerate(self.devices):
            # Move input tensors to the correct device
            input_ids_split[i] = input_ids_split[i].to(dev)
            attention_mask_split[i] = attention_mask_split[i].to(dev)

            # Ensure the model is on the correct device
            model_on_device = self.model.to(dev)

            # Perform the forward pass on the model
            with torch.no_grad():
                output = model_on_device(input_ids=input_ids_split[i], attention_mask=attention_mask_split[i])

            outputs.append(output.logits)

        # Ensure that all outputs are on the same device (move them to the same device, e.g., cuda:0)
        outputs = [output.to(self.devices[0]) for output in outputs]  # Move all to cuda:0 or chosen device

        # Concatenate the results from all GPUs
        return torch.cat(outputs, dim=0)

    def parameters(self):
        return self.model.parameters()

# Define devices and move model to devices
device_ids = [0, 1]  # Specify the GPUs you want to use
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
parallel_model = ManualDataParallel(model, device_ids)

# Create a batch of input texts
input_texts = ["Hello, world!", "How are you?", "What is your name?", "Tell me a joke.", "Give me a quote."] * 1000  # Large batch
inputs = tokenizer(input_texts, padding=True, return_tensors="pt")

# Move input tensors to the same device as the model
input_ids = inputs.input_ids.to(device)
attention_mask = inputs.attention_mask.to(device)

# Use model.generate for text generation (ManualDataParallel will distribute the batch across GPUs)
generated_ids = parallel_model.forward(input_ids, attention_mask)

# Decode and print generated text
generated_texts = [tokenizer.decode(ids, skip_special_tokens=True) for ids in generated_ids]
for i, generated_text in enumerate(generated_texts):
    print(f"Generated text {i+1}: {generated_text}")
