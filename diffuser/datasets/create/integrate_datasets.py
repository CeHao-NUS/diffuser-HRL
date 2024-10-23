

'''
Load more than one datasets, then integrates them

'''

import h5py
import os
import json
import numpy as np
import gym
from tqdm import tqdm
import itertools

from diffuser.datasets.d4rl import load_environment


def process_env_rewards(reward):
    out_rewrads = np.ones_like(reward)
    print('length of env file:', len(reward))
    return out_rewrads

def process_local_rewards(reward):
    # set failure reward = -1
    out_rewrads = np.ones_like(reward)
    fail_index = np.where(reward < 0.1)
    out_rewrads[fail_index] = -1

    print('length of local file:', len(reward))
    print('failure rate:', len(fail_index[0])/len(reward))
    return out_rewrads


def reset_data():
    return {
        'observations': [],
        'actions': [],
        'terminals': [],
        'timeouts': [],
        'rewards': [],
    }

def append_data(data, s, a, done, timeouts, reward):
    data['observations'].append(s.tolist())
    data['actions'].append(a.tolist())
    data['terminals'].append(done.tolist())
    data['timeouts'].append(timeouts.tolist())
    data['rewards'].append(reward.tolist())  # Add rewards to dataset

def flatten_data(data):
        return {
        'observations': list(itertools.chain(*data['observations'])),
        'actions': list(itertools.chain(*data['actions'])),
        'terminals': list(itertools.chain(*data['terminals'])),
        'timeouts': list(itertools.chain(*data['timeouts'])),
        'rewards': list(itertools.chain(*data['rewards'])),
    }
# ===========================================================================
def maze2d_set_terminals(env):
    env = load_environment(env) if type(env) == str else env
    goal = np.array(env._target)
    threshold = 0.5

    def _fn(dataset):
        xy = dataset['observations'][:,:2]
        distances = np.linalg.norm(xy - goal, axis=-1)
        at_goal = distances < threshold
        timeouts = np.zeros_like(dataset['timeouts'])

        ## timeout at time t iff
        ##      at goal at time t and
        ##      not at goal at time t + 1
        timeouts[:-1] = at_goal[:-1] * ~at_goal[1:]

        timeout_steps = np.where(timeouts)[0]
        path_lengths = timeout_steps[1:] - timeout_steps[:-1]

        print(
            f'[ utils/preprocessing ] Segmented {env.name} | {len(path_lengths)} paths | '
            f'min length: {path_lengths.min()} | max length: {path_lengths.max()}'
        )

        dataset['timeouts'] = timeouts
        return dataset

    return _fn

def read_env_dataset(data_all, env_name):
    # env = gym.make(env_name)
    env = load_environment(env_name)

    dataset = env.get_dataset()
    dataset = maze2d_set_terminals(env)(dataset)

    dataset['rewards'] = process_env_rewards(dataset['rewards'])
    append_data(data_all, dataset['observations'], dataset['actions'], dataset['terminals'], dataset['timeouts'], dataset['rewards'])

# =============================================================================
def get_keys(h5file):
    keys = []

    def visitor(name, item):
        if isinstance(item, h5py.Dataset):
            keys.append(name)

    h5file.visititems(visitor)
    return keys

def get_dataset(h5path=None):

        data_dict = {}
        with h5py.File(h5path, 'r') as dataset_file:
            for k in tqdm(get_keys(dataset_file), desc="load datafile"):
                try:  # first try loading as an array
                    data_dict[k] = dataset_file[k][:]
                except ValueError as e:  # try loading as a scalar
                    data_dict[k] = dataset_file[k][()]
        return data_dict


def read_local_dataset(data_all, local_name):
    data_dict = get_dataset(local_name)
    data_dict['rewards'] = process_local_rewards(data_dict['rewards'])
    append_data(data_all, data_dict['observations'], data_dict['actions'], data_dict['terminals'], data_dict['timeouts'], data_dict['rewards'])


def save_data(data_dir, file_name, all_data):
    dir_name = ''
    os.makedirs(os.path.join(data_dir, dir_name), exist_ok=True)
    file_name = os.path.join(data_dir, dir_name, file_name)

    # Save the full concatenated dataset into a single HDF5 file
    with h5py.File(file_name, "w") as f:
        # Save the core data arrays from all_data dictionary
        for key, data in all_data.items():
            f.create_dataset(key, data=np.array(data))

    print(f"Dataset saved at {file_name}")


def integrate(env_list=[], local_list=[], output_dir=''):
    data_all = reset_data()

    for env_name in env_list:
        read_env_dataset(data_all, env_name)

    for local_name in local_list:
        read_local_dataset(data_all, local_name)


    data_all = flatten_data(data_all)
    save_data(output_dir, 'integrated.h5', data_all)
    
if __name__ == '__main__':
    # env_list = ['maze2d-medium-v1']
    env_list = []
    local_list = ['temp_datasets/med_single_maze2d.hdf5']
    output_dir = './temp_datasets'
    integrate(env_list, local_list, output_dir)
