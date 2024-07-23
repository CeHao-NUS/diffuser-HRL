import pdb

import diffuser.sampling as sampling
import diffuser.utils as utils
import numpy as np
from os.path import join
import json
#-----------------------------------------------------------------------------#
#----------------------------------- setup -----------------------------------#
#-----------------------------------------------------------------------------#

class Parser(utils.Parser):
    dataset: str = 'maze2d-umaze-v1'
    config: str = 'config.mazenew.maze2d'

args = Parser().parse_args('plan')

#-----------------------------------------------------------------------------#
#---------------------------------- loading ----------------------------------#
#-----------------------------------------------------------------------------#

## load diffusion model and value function from disk
diffusion_experiment = utils.load_diffusion(
    args.loadbase, args.dataset, args.diffusion_loadpath,
    epoch=args.diffusion_epoch, seed=args.seed,
)

value_experiment = utils.load_diffusion(
    args.loadbase, args.dataset, args.value_loadpath,
    epoch=args.value_epoch, seed=args.seed,
)

# diffusion = diffusion_experiment.ema
dataset = diffusion_experiment.dataset
# renderer = diffusion_experiment.renderer


## initialize value guide
value_function = value_experiment.ema
guide_config = utils.Config(args.guide, model=value_function, verbose=False)
guide = guide_config()



# tests

target = [2.0, 3.0]
goal = [*target, 0, 0]
action = [0.0, 0.0]

goal_norm = dataset.normalizer.normalize(goal, 'observations')
action_norm = dataset.normalizer.normalize(action, 'actions')
goal_norm = np.array(goal_norm, dtype=np.float32)
action_norm = np.array(action_norm, dtype=np.float32)
import torch
x_goal = np.concatenate([action_norm, goal_norm], axis=0)
goal = torch.tensor(x_goal, device=args.device)

guide.set_goal(goal)


# for t0 in reversed(range(32)):
t0 = 0
point_list = [[2.0, 3.0], [2.5, 3.0], [2.0, 3.5], [1.5, 3.0], [2.0, 2.5], [2.5, 3.5], [1.5, 2.5], [1.5, 3.5], [2.5, 2.5]]

for pt in point_list:
    # x = torch.tensor([[0.0, 0.0, *(pt),  0.0, 0.0]], device=args.device)
    # observation = torch.tensor([*pt, 0.0, 0.0], device=args.device)
    # action = torch.tensor([0.0, 0.0], device=args.device)

    observation = np.array([*pt, 0.0, 0.0], dtype=np.float32)
    action = np.array([0.0, 0.0], dtype=np.float32)
    obs_norm = dataset.normalizer.normalize(observation, 'observations')
    act_norm = dataset.normalizer.normalize(action, 'actions')

    obs_norm = torch.tensor(obs_norm, device=args.device)
    act_norm = torch.tensor(act_norm, device=args.device)

    x = torch.cat([act_norm, obs_norm], dim=0)

    x = x.repeat(1, 32, 1)

    cond = {}
    t = torch.tensor([t0], device=args.device)
    y, grad = guide.gradients(x, cond, t)

    # print("====================== t0", t0)
    print('+=============== pt', pt)
    print('y', y.detach().cpu().numpy())
    print('grad', grad[0, -1,:].detach().cpu().numpy())