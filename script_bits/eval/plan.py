
import pdb

import diffuser.sampling as sampling
import diffuser.utils as utils
import numpy as np
from os.path import join
import json

import torch

#-----------------------------------------------------------------------------#
#----------------------------------- setup -----------------------------------#
#-----------------------------------------------------------------------------#

class Parser(utils.Parser):
    dataset: str = 'tamp_easy'
    config: str = 'config_bits.diffusion.plan_diff'

args = Parser().parse_args('plan')


#-----------------------------------------------------------------------------#
#---------------------------------- loading ----------------------------------#
#-----------------------------------------------------------------------------#

## load diffusion model and value function from disk
diffusion_experiment = utils.load_diffusion(
    args.loadbase, args.dataset, args.diffusion_loadpath,
    epoch=args.diffusion_epoch, seed=args.seed,
    device=args.device
)


diffusion = diffusion_experiment.ema
dataset = diffusion_experiment.dataset
renderer = diffusion_experiment.renderer

# reset the dataset.
# dataset.update_datset(data_dir="/home/crtie/.d4rl/datasets/tamp_p0.1_n1000/dataset.txt")


if args.value_loadpath is not None:

    value_experiment = utils.load_diffusion(
        args.loadbase, args.dataset, args.value_loadpath,
        epoch=args.value_epoch, seed=args.seed,
        device=args.device
    )

    ## ensure that the diffusion model and value function are compatible with each other
    # utils.check_compatibility(diffusion_experiment, value_experiment)

    ## initialize value guide
    value_function = value_experiment.ema
    guide_config = utils.Config(args.guide, model=value_function, verbose=False)
    guide = guide_config()

    ## policies are wrappers around an unconditional diffusion model and a value guide
    policy_config = utils.Config(
        args.policy,
        guide=guide,
        scale=args.scale,
        diffusion_model=diffusion,
        normalizer=dataset.normalizer,
        preprocess_fns=args.preprocess_fns,
        ## sampling kwargs
        sample_fn=sampling.n_step_guided_p_sample,
        n_guide_steps=args.n_guide_steps,
        t_stopgrad=args.t_stopgrad,
        scale_grad_by_std=args.scale_grad_by_std,
        verbose=False,
    )

    # logger = logger_config()
    policy = policy_config()
else:
    from diffuser.guides.bit_policy import BitPolicy
    policy = BitPolicy(diffusion)



#---------------------------------- main loop ----------------------------------#
from diffuser.utils.arrays import batch_to_device, to_np
from diffuser.utils.rendering import StateTrajRenderer

# 0. get the cond from dataset
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1024)
render = StateTrajRenderer(tokenizer_save_path='./custom_tokenizer', n_bits=dataset.n_bits)


# remove args.save_path_dir file

import os
if os.path.exists(args.save_path_dir):
    os.remove(args.save_path_dir)

for batch in dataloader:
    batch = batch_to_device(batch, device=args.device)


    # 1. use the policy to run sample for all data. save
    samples = diffusion(batch.conditions, batch_size=len(batch.conditions[0]))
    observations = to_np(samples.trajectories)

    # parse to text, save to local text file

    render.composite(args.save_path_dir, observations)


# 2. call the dynamics and calculate the results

from diffuser.datasets.bits_fun.gen_dataset.dynamics import eval_env

all_length, converted_stage_action_num = eval_env(args.save_path_dir)


# 3. 

a = 1




## save result as a json file
# json_path = join(args.savepath, 'rollout.json')
# json_data = {'score': score, 'step': t, 'return': total_reward, 'term': terminal,
#     'epoch_diffusion': diffusion_experiment.epoch}
# json.dump(json_data, open(json_path, 'w'), indent=2, sort_keys=True)

# # save samples to pickle
# samples = {'observations': samples.observations, 'actions': samples.actions}
# import pickle
# with open(join(args.savepath, 'samples.pkl'), 'wb') as f:
#     pickle.dump(samples, f)

