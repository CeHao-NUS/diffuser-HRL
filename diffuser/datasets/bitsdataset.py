import numpy as np
from transformers import AutoTokenizer

from collections import namedtuple
import torch
import pdb


import os
import diffuser
def load_custom_texts(file_path='datasets/bit_dataset.txt'):

    based_path = os.path.join(os.path.dirname(diffuser.__file__),file_path)

    with open(based_path, 'r') as file:
        lines = file.readlines()

    tasks = []
    task = ""
    for line in lines:
        stripped_line = line.strip()
        
        # Check if the line starts a new task (assuming each new task starts with "Task x:")
        if stripped_line.startswith("Task"):
            if task:
                tasks.append(task.strip())  # Append the previous task to the list
            task = stripped_line  # Start a new task
        else:
            task += " " + stripped_line  # Append step to the current task

    if task:
        tasks.append(task.strip())  # Append the last task

    return tasks


# Define int2bits and bits2int as before
def int2bits(x, n, out_dtype=None):
    """Convert an integer x in (...) into bits in (..., n)."""
    x = np.right_shift(np.expand_dims(x, -1), np.arange(n))
    x = np.mod(x, 2)
    if out_dtype and out_dtype != x.dtype:
        x = x.astype(out_dtype)
    return x

def bits2int(x, out_dtype):
    """Converts bits x in (..., n) into an integer in (...)."""
    x = x.astype(out_dtype)
    x = np.sum(x * (2 ** np.arange(x.shape[-1])), axis=-1)
    return x

def bits2range(x):
    # convert the 0 in x to -1
    return 2 * x - 1

def range2bits(x):
    # set in x, if x > 0, set as 1, else 0
    y = x.copy()
    y[x>=0] = 1
    y[x<0] = 0
    return y

def text_to_bits(text, tokenizer, n_bits=16):
    """Tokenize text and convert tokens to binary bits."""
    # Tokenize without special tokens
    tokens = tokenizer(text, add_special_tokens=False, return_tensors="np")["input_ids"].squeeze()
    # print('tokens:', tokens)
    bits = int2bits(tokens, n=n_bits, out_dtype=np.int32)
    bits = bits2range(bits)
    return bits

def bits_to_text(bits, tokenizer, n_bits=16):
    """Convert binary bits back to text using tokenizer."""
    # Convert bits to token IDs
    bits = range2bits(bits)
    token_ids = bits2int(bits, out_dtype=np.int32)
    
    # Decode token IDs to text, skip special tokens to ensure clean output
    text = tokenizer.decode(token_ids, skip_special_tokens=True)
    return text


Batch = namedtuple('Batch', 'trajectories conditions')


class Normalizer:
    def normalize(self, text, type):
        return text

    def unnormalize(self, bits, type):
        # return bits_to_text(bits)
        return bits

class BitDataset(torch.utils.data.Dataset):

    def __init__(self, text_data_dir='datasets/bit_dataset.txt', n_bits=16, horizon=64, observation_dim=16, action_dim=0):
        self.text_data = load_custom_texts(text_data_dir)
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self.n_bits = n_bits
        self.bits =[text_to_bits(text, self.tokenizer, n_bits) for text in self.text_data]
        self.horizon = horizon

        self.observation_dim = observation_dim
        self.action_dim = action_dim

        self.normalizer = Normalizer()  

    def __len__(self):
        return len(self.bits)
    
    def get_conditions(self, observations):
        '''
            condition on current observation for planning
        '''
        return {0: observations[0]}
    
    def __getitem__(self, idx):
        # print('idx:', idx)
        observations = self.bits[idx][:self.horizon]
        # to float
        observations = observations.astype(np.float32)

        # text = self.text_data[idx]
    
        conditions = self.get_conditions(observations)
        # trajectories = np.concatenate([actions, observations], axis=-1)
        trajectories = observations
        batch = Batch(trajectories, conditions)

        assert observations.shape == (32, 16)
        return batch
    
    def decode_bit2text(self, bits):
        return bits_to_text(bits, self.tokenizer, self.n_bits)


if __name__ == "__main__":

    file_dir = 'datasets/bit_dataset.txt'
    dataset = BitDataset(file_dir)