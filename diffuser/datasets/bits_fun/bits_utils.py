

import os
import diffuser
# ========================= read the texts
def load_custom_texts(file_path, remove_newline=False):

    based_path = os.path.join(os.path.dirname(diffuser.__file__),file_path)

    with open(based_path, 'r') as file:
        lines = file.readlines()

    texts = []
    text = ""
    for line in lines:
        if not text: # start a new line
            if line == "\n":
                continue
            text = text + line
        else:
            if line == "\n":
                texts.append(text)
                text = ""
            else:
                text = text + line
    
    if remove_newline: 
        return [remove_next_line(text) for text in texts] 
    else:
        return texts

def remove_next_line(text):
    return text.replace("\n", "")

# ========================= create tokenizer
from collections import Counter
import re
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Sequence, Whitespace, Punctuation
from transformers import PreTrainedTokenizerFast

def create_tokenizer(texts, save_path="./custom_tokenizer"):

    # Step 2: Build the vocabulary from the article (with punctuation split)
    # Tokenize the article
    split_tokens = re.findall(r'\w+|[^\w\s]', texts)  # Split on words and punctuation
    word_freq = Counter(split_tokens)

    # Create a vocabulary with unique tokens and add special tokens
    vocab = {word: idx for idx, (word, _) in enumerate(word_freq.items())}
    vocab["[PAD]"] = len(vocab)  # Padding token
    vocab["[UNK]"] = len(vocab)  # Unknown token

    # Step 3: Create a WordLevel tokenizer
    tokenizer = Tokenizer(WordLevel(vocab=vocab, unk_token="[UNK]"))

    # Use a combination of whitespace and punctuation splitting
    tokenizer.pre_tokenizer = Sequence([Whitespace(), Punctuation()])

    # Step 4: Wrap in PreTrainedTokenizerFast for compatibility
    tokenizer_fast = PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="[UNK]", pad_token="[PAD]")

    # Save the tokenizer
    tokenizer_fast.save_pretrained(save_path)


# =========================  a class to parse the text
def convert_state_trajectory(text_data, no_first_line=True):
    # 1. first row is task description
    # 2. the rest are steps

    # create a dict, key is the task, value is the steps {task: [['close', 'infridge'], ['open', 'onpan']]}
    task_dict = {}
    for idx, texts in enumerate(text_data):
        lines = texts.strip().split("\n")
        if no_first_line:
            task = f"task_{idx}"
            steps = lines
        else:
            task = lines[0]
            steps = lines[1:]
        task_dict[task] = []

        for i, step in enumerate(steps):
            if "." in step:
                step_no_index = step.split(".")[1].strip()
            else:
                step_no_index = step
            # seperate by whitespace 
            simplified_state = [item.split()[-1] for item in step_no_index.split(' ')]
            task_dict[task].append(simplified_state)

    return task_dict

def flatten_and_concatenate(nested_list):
    result = []
    for element in nested_list:
        if isinstance(element, list):
            # Recursively flatten nested lists
            result.extend(flatten_and_concatenate(element))
        elif isinstance(element, str):
            # Append strings to result
            result.append(element)
    return result


def convert_tokenized_state_traj(state_traj, tokenizer, n_bits=16):
    
    dataset = []

    for task, steps in state_traj.items():
        state_traj = []
        for step in steps:
            state = []
            for obj in step:
                state.append(text_to_bits(obj, tokenizer, n_bits))
            state_traj.append(np.concatenate(state))
        dataset.append(np.array(state_traj))

    return dataset


def divide_list(original_list, k):
    # Split the list into sublists of size k
    divided_list = [original_list[i:i + k] for i in range(0, len(original_list), k)]
    return divided_list

import numpy as np
# ============================= utils of converter

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


def bits_to_text_customize(bits, tokenizer, n_bits=16):
        # Convert bits to token IDs
    bits = range2bits(bits)
    token_ids = bits2int(bits, out_dtype=np.int32)
    
    # Decode token IDs to text, skip special tokens to ensure clean output
    # text = tokenizer.convert_ids_to_tokens(token_ids, skip_special_tokens=True)
    text = tokenizer.convert_ids_to_tokens(token_ids)
    return text