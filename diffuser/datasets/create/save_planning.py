
# to save the planning results into datasets

'''
In planning, save the planning results to a numpy file
Then convert them to h5py, with special reward
'''

import os
import json
import numpy as np
import pickle
from tqdm import tqdm
import h5py
import itertools

def reset_data():
    return {
        'observations': [],
        'actions': [],
        'terminals': [],
        'timeouts': [],
        'rewards': [],
    }

def append_data(data, s, a, done, timeout, reward):
    data['observations'].append(s)
    data['actions'].append(a)
    data['terminals'].append(done)
    data['timeouts'].append(timeout)
    data['rewards'].append(reward)  # Add rewards to dataset

def add_dummy_data(data_list):
    data_list.append(data_list[-1])

def flatten_data(data):
        return {
        'observations': list(itertools.chain(*data['observations'])),
        'actions': list(itertools.chain(*data['actions'])),
        'terminals': list(itertools.chain(*data['terminals'])),
        'timeouts': list(itertools.chain(*data['timeouts'])),
        'rewards': list(itertools.chain(*data['rewards'])),
    }

def plan2dataset(data, samples, scores):
    obsevations = samples['observations'][0].tolist()
    actions = samples['actions'][0].tolist()

    n = len(obsevations)
    done = np.full(n, False).tolist()
    timeout = np.full(n, False).tolist()
    rewards = np.full(n, scores).tolist()
    add_dummy_data(obsevations)
    add_dummy_data(actions)
    add_dummy_data(done)
    add_dummy_data(timeout)
    add_dummy_data(rewards)

    done[-1] = True
    timeout[-1] = True
    append_data(data, obsevations, actions, done, timeout, rewards)


def read_form_plan_dataset(subfolder, env):
    # similar to eval, iteratively read the dataset folder
    
    base_dir = f'./logs/{env}/plans/{subfolder}'
    
    rollout_name = 'rollout.json'
    sample_name = 'samples.pkl'

    # number of files in base_dir
    eval_idx_names = os.listdir(base_dir)

    data_all = reset_data()

    for file_name in tqdm(eval_idx_names):
        rollout_path = os.path.join(base_dir, file_name, rollout_name)
        sample_path = os.path.join(base_dir, file_name, sample_name)

        # load json file
        with open(rollout_path, 'r') as f:
            data = json.load(f)
            score = data['score']   

        # if score < 0.1: # store the failure cases

        # load samples
        with open(sample_path, 'rb') as f:
            samples = pickle.load(f)

        plan2dataset(data_all, samples, score)


    return flatten_data(data_all)


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


if __name__ == "__main__":

    env_list = ['maze2d-umaze-v1', 'maze2d-medium-v1', 'maze2d-large-v1', 'maze2d-testbig-v0']
    env = env_list[1]

    subfolder = 'single_save_H256_T256_d0.99_b1_condTrue'
    data_dir = './temp_datasets'
    file_name = 'med_single_maze2d.hdf5'
    
    data = read_form_plan_dataset(subfolder, env)
    save_data(data_dir, file_name, data)

