from collections import namedtuple
import numpy as np
import torch
import pdb

from .preprocessing import get_preprocess_fn
from .d4rl import load_environment, sequence_dataset
from .normalization import DatasetNormalizer
from .buffer import ReplayBuffer


Batch = namedtuple('Batch', 'trajectories conditions')
ValueBatch = namedtuple('ValueBatch', 'trajectories conditions values')


class SequenceDataset(torch.utils.data.Dataset):

    def __init__(self, env='hopper-medium-replay', horizon=64,
        normalizer='LimitsNormalizer', preprocess_fns=[], max_path_length=1000,
        max_n_episodes=10000, termination_penalty=0, use_padding=True, seed=None, **kwargs):
        self.preprocess_fn = get_preprocess_fn(preprocess_fns, env)
        self.env = env = load_environment(env)
        self.env.seed(seed)
        self.horizon = horizon
        self.max_path_length = max_path_length
        self.use_padding = use_padding
        itr = sequence_dataset(env, self.preprocess_fn)

        fields = ReplayBuffer(max_n_episodes, max_path_length, termination_penalty)
        for i, episode in enumerate(itr):
            fields.add_path(episode)
        fields.finalize()

        self.normalizer = DatasetNormalizer(fields, normalizer, path_lengths=fields['path_lengths'])
        self.indices = self.make_indices(fields.path_lengths, horizon)

        self.observation_dim = fields.observations.shape[-1]
        self.action_dim = fields.actions.shape[-1]
        self.fields = fields
        self.n_episodes = fields.n_episodes
        self.path_lengths = fields.path_lengths
        self.normalize()

        print(fields)
        # shapes = {key: val.shape for key, val in self.fields.items()}
        # print(f'[ datasets/mujoco ] Dataset fields: {shapes}')

    def normalize(self, keys=['observations', 'actions']):
        '''
            normalize fields that will be predicted by the diffusion model
        '''
        for key in keys:
            array = self.fields[key].reshape(self.n_episodes*self.max_path_length, -1)
            normed = self.normalizer(array, key)
            self.fields[f'normed_{key}'] = normed.reshape(self.n_episodes, self.max_path_length, -1)

    def make_indices(self, path_lengths, horizon):
        '''
            makes indices for sampling from dataset;
            each index maps to a datapoint
        '''
        indices = []
        for i, path_length in enumerate(path_lengths):
            max_start = min(path_length - 1, self.max_path_length - horizon)
            if not self.use_padding:
                max_start = min(max_start, path_length - horizon)
            for start in range(max_start):
                end = start + horizon
                indices.append((i, start, end))
        indices = np.array(indices)
        return indices

    def get_conditions(self, observations):
        '''
            condition on current observation for planning
        '''
        return {0: observations[0]}

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx, eps=1e-4):
        path_ind, start, end = self.indices[idx]

        observations = self.fields.normed_observations[path_ind, start:end]
        actions = self.fields.normed_actions[path_ind, start:end]

        conditions = self.get_conditions(observations)
        trajectories = np.concatenate([actions, observations], axis=-1)
        batch = Batch(trajectories, conditions)
        return batch


class GoalDataset(SequenceDataset):

    def get_conditions(self, observations):
        '''
            condition on both the current observation and the last observation in the plan
        '''
        return {
            0: observations[0],
            self.horizon - 1: observations[-1],
        }
    
class DummyGoalDataset(SequenceDataset):
    def get_conditions(self, observations):
        return {
            0: observations[0],
        }
    
class BatchGaolDataset(SequenceDataset):
    def get_conditions(self, observations):
        '''
            condition on both the current observation and the last observation in the plan
        '''
        return {
            (0,0): observations[0],
            (0, self.horizon - 1): observations[-1],
        }

class ValueDataset(SequenceDataset):
    '''
        adds a value field to the datapoints for training the value function
    '''

    def __init__(self, *args, discount=0.99, normed=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.discount = discount
        self.discounts = self.discount ** np.arange(self.max_path_length)[:,None]
        self.normed = False
        if normed:
            self.vmin, self.vmax = self._get_bounds()
            self.normed = True

    def _get_bounds(self):
        print('[ datasets/sequence ] Getting value dataset bounds...', end=' ', flush=True)
        vmin = np.inf
        vmax = -np.inf
        for i in range(len(self.indices)):
            value = self.__getitem__(i).values.item()
            vmin = min(value, vmin)
            vmax = max(value, vmax)
        print('✓')
        return vmin, vmax

    def normalize_value(self, value):
        ## [0, 1]
        normed = (value - self.vmin) / (self.vmax - self.vmin)
        ## [-1, 1]
        normed = normed * 2 - 1
        return normed

    def __getitem__(self, idx):
        batch = super().__getitem__(idx)
        path_ind, start, end = self.indices[idx]
        rewards = self.fields['rewards'][path_ind, start:]
        discounts = self.discounts[:len(rewards)]
        value = (discounts * rewards).sum()
        if self.normed:
            value = self.normalize_value(value)
        value = np.array([value], dtype=np.float32)
        value_batch = ValueBatch(*batch, value)
        return value_batch


class GoalValueDataset(ValueDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.min_horizon = 2
    
    def __getitem__(self, idx):
        # batch = super().__getitem__(idx)
        # path_ind, start, end = self.indices[idx]

        path_ind, start, end = self.indices[idx]
        horizon = self.horizon

        new_length = np.random.choice(range(self.min_horizon, horizon))
        repeats = horizon - new_length
        new_end = start + new_length

        observations = self.fields.normed_observations[path_ind, start:new_end]
        actions = self.fields.normed_actions[path_ind, start:new_end]

        # repeat the last observation until end
        # observations = np.concatenate([observations, np.repeat(observations[-1, np.newaxis, :], repeats, axis=0)], axis=0)
        observations = np.concatenate([observations[:1], observations[-1:]], axis=0)
        
        # zero_actions = np.zeros_like(actions[-1])
        # actions = np.concatenate([actions, np.repeat(zero_actions[np.newaxis, :], repeats, axis=0)], axis=0)
        actions = np.concatenate([actions[:1], actions[-1:]], axis=0)
        actions = np.zeros_like(actions)

        conditions = self.get_conditions(observations)
        trajectories = np.concatenate([actions, observations], axis=-1)


        rewards = self.fields['rewards'][path_ind, start:new_end]
        rewards = np.ones_like(rewards) * -1 # make the steps to be negative

        discounts = self.discounts[:len(rewards)]
        value = (discounts * rewards).sum()
        
        if self.normed:
            value = self.normalize_value(value)

        value = np.array([value], dtype=np.float32)
        value_batch = ValueBatch(trajectories, conditions, value)
        return value_batch

    
class VarHDataset1(GoalDataset):
    def __init__(self, *args, discount=0.99, normed=False, min_horizon=1, **kwargs):
        self.min_horizon = min_horizon
        super().__init__(*args, **kwargs)
        self.discount = discount
        self.discounts = self.discount ** np.arange(self.max_path_length)[:,None]
        self.normed = normed

    def make_indices(self, path_lengths, horizon):
        '''
            makes indices for sampling from dataset;
            each index maps to a datapoint
        '''
        indices = []
        for i, path_length in enumerate(path_lengths):
            # the true min horizon is self.min_horizon
            max_start = min(path_length - 1, self.max_path_length - self.min_horizon)

            if not self.use_padding:
                max_start = min(max_start, path_length - self.min_horizon)

            for start in range(max_start):
                shorten_horizon = np.random.choice(range(self.min_horizon, horizon)) # shorten the horizon
                end = start + shorten_horizon
                # now end should be less than path_length
                end = min(end, path_length-1)
                indices.append((i, start, end))
        indices = np.array(indices)
        return indices

    def __getitem__(self, idx):

        # 1. get intermediate point
        # 2. segment and fill with last point
        # 3. change conditions

        path_ind, start, end = self.indices[idx]

        shorten_length = end - start
        max_horizon = self.horizon

        repeats = max_horizon - shorten_length

        observations = self.fields.normed_observations[path_ind, start:end]
        actions = self.fields.normed_actions[path_ind, start:end]

        # repeat the last observation until end
        observations = np.concatenate([observations, np.repeat(observations[-1, np.newaxis, :], repeats, axis=0)], axis=0)
        
        zero_actions = np.zeros_like(actions[-1])
        actions = np.concatenate([actions, np.repeat(zero_actions[np.newaxis, :], repeats, axis=0)], axis=0)

        conditions = self.get_conditions(observations)
        trajectories = np.concatenate([actions, observations], axis=-1)

        batch = Batch(trajectories, conditions)
        return batch
    

class VarHDataset2(GoalDataset):
    def __init__(self, *args, discount=0.99, normed=False, min_horizon=1, **kwargs):
        self.min_horizon = min_horizon
        super().__init__(*args, **kwargs)
        self.discount = discount
        self.discounts = self.discount ** np.arange(self.max_path_length)[:,None]
        self.normed = normed

    def __getitem__(self, idx):


        # 1. get intermediate point
        # 2. segment and fill with last point
        # 3. change conditions

        path_ind, start, end = self.indices[idx]
        horizon = self.horizon

        new_length = np.random.choice(range(self.min_horizon, horizon))
        repeats = horizon - new_length
        new_end = start + new_length

        observations = self.fields.normed_observations[path_ind, start:new_end]
        actions = self.fields.normed_actions[path_ind, start:new_end]

        # repeat the last observation until end
        observations = np.concatenate([observations, np.repeat(observations[-1, np.newaxis, :], repeats, axis=0)], axis=0)
        
        zero_actions = np.zeros_like(actions[-1])
        actions = np.concatenate([actions, np.repeat(zero_actions[np.newaxis, :], repeats, axis=0)], axis=0)

        conditions = self.get_conditions(observations)
        trajectories = np.concatenate([actions, observations], axis=-1)

        batch = Batch(trajectories, conditions)
        return batch

class VarHValueDataset(VarHDataset1):


    def __getitem__(self, idx):
        path_ind, start, end = self.indices[idx]
        horizon = self.horizon

        new_length = np.random.choice(range(self.min_horizon, horizon))
        repeats = horizon - new_length
        new_end = start + new_length

        observations = self.fields.normed_observations[path_ind, start:new_end]
        actions = self.fields.normed_actions[path_ind, start:new_end]

        # repeat the last observation until end
        observations = np.concatenate([observations, np.repeat(observations[-1, np.newaxis, :], repeats, axis=0)], axis=0)
        
        zero_actions = np.zeros_like(actions[-1])
        actions = np.concatenate([actions, np.repeat(zero_actions[np.newaxis, :], repeats, axis=0)], axis=0)

        conditions = self.get_conditions(observations)
        trajectories = np.concatenate([actions, observations], axis=-1)

        # reward
        rewards = self.fields['rewards'][path_ind, start:new_end]
        rewards = np.zeros_like(rewards)  # make the steps to be negative
        rewards[:, :new_length] = -1

        discounts = self.discounts[:len(rewards)]
        value = (discounts * rewards).sum()
        value = np.array([value], dtype=np.float32)
        value_batch = ValueBatch(trajectories, conditions, value)

        return value_batch
    

class JointValueDataset(GoalDataset):
    def __init__(self, *args, discount=0.99, normed=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.discount = discount
    
    def __getitem__(self, idx):
        path_ind, start, end = self.indices[idx]

        observations = self.fields.normed_observations[path_ind, start:end]
        actions = self.fields.normed_actions[path_ind, start:end]

        # conditions = self.get_conditions(observations)
        conditions = {} # should not use hard conditions
        

        # create s0' = s0+noise, sT' = sT+noise
        # add noise to the first and last observations
        s0 = observations[0, :]
        sT = observations[-1, :] # (4,)
        noise0 = np.random.normal(0, 0.2, s0.shape).astype(np.float32)
        noiseT = np.random.normal(0, 0.2, sT.shape).astype(np.float32)
        s0_prime = s0 + noise0
        sT_prime = sT + noiseT
        observations = np.concatenate([s0_prime[np.newaxis, :], observations, sT_prime[np.newaxis, :]], axis=0)

        zero_actions = np.zeros_like(actions[0, :])
        actions = np.concatenate([zero_actions[np.newaxis, :], actions, zero_actions[np.newaxis, :]], axis=0)
        trajectories = np.concatenate([actions, observations], axis=-1)

        norm_noise0 = np.linalg.norm(noise0)
        norm_noiseT = np.linalg.norm(noiseT)

        value = norm_noise0 + norm_noiseT

        value = np.array([value], dtype=np.float32)
        value_batch = ValueBatch(trajectories, conditions, value)

        return value_batch
    
