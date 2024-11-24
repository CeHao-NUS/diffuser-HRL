import numpy as np
from transformers import AutoTokenizer

from collections import namedtuple
import torch
import pdb

from diffuser.datasets.bits_fun.bits_utils import *



Batch = namedtuple('Batch', 'trajectories conditions')


class Normalizer:
    def __init__(self, observation_dim=0, action_dim=0):
        self.observation_dim = observation_dim
        self.action_dim = action_dim

    def normalize(self, text, type):
        return text

    def unnormalize(self, bits, type):
        # return bits_to_text(bits)
        return bits


class StateTrajBitDataset(torch.utils.data.Dataset):

    def __init__(self, data_dir='', horizon=12, n_bits=4, n_objs=9, set_tokenizer=False, tokenizer_save_path='./custom_tokenizer',
                 cond_index=[0], **kwargs):
        # 1. read dataset
        text_data = load_custom_texts(data_dir) # a list of strings

        # pre-process, remove hyphen
        text_data = [text.replace("-", "") for text in text_data]
    
        # 2. form the state trajectory
        state_trajectory = convert_state_trajectory(text_data)

        # 3. create tokenizer (can choose)
        if set_tokenizer:
            all_data_texts = flatten_and_concatenate ( list(state_trajectory.values()) )
            all_data_texts = " ".join(all_data_texts)
            create_tokenizer(all_data_texts, tokenizer_save_path)
        
        self.tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_save_path)
        print('self.tokenizer', self.tokenizer.vocab_size)

        # 4. create the dataset / a list of np.array / [ (horizon, n_bits * n_objs), [], []]
        self.dataset = convert_tokenized_state_traj(state_trajectory, self.tokenizer, n_bits)

        self.padding_token = text_to_bits('[PAD]', self.tokenizer, n_bits)
        # repeat n_objs times
        self.padding_token = np.tile(self.padding_token, (n_objs))

        self.horizon = horizon
        self.n_bits = n_bits
        self.n_objs = n_objs

        self.observation_dim = n_bits * n_objs
        self.action_dim = 0
        self.cond_index = cond_index

        self.normalizer = Normalizer(self.observation_dim, self.action_dim)

    def __len__(self):
        return len(self.dataset)
    
    def get_conditions(self, observations):
        # return {0: observations[0],
        #         self.horizon -1 : observations[-1]}

        return {i: observations[i] for i in self.cond_index}
    
    def __getitem__(self, idx):
        observations = self.dataset[idx]

        # if length < self.horizon, pad with the last observation
        if len(observations) < self.horizon:
            observations = np.concatenate([observations, np.tile(self.padding_token, (self.horizon - len(observations), 1))])
        else:
            observations = observations[:self.horizon]

        observations = observations.astype(np.float32)

        conditions = self.get_conditions(observations)
        trajectories = observations
        batch = Batch(trajectories, conditions)

        return batch


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

        # assert observations.shape == (32, 16)
        return batch
    
    def decode_bit2text(self, bits):
        return bits_to_text(bits, self.tokenizer, self.n_bits)


if __name__ == "__main__":

    file_dir = 'datasets/bits_fun/bit_dataset.txt'
    dataset = StateTrajBitDataset(file_dir, set_tokenizer=True)