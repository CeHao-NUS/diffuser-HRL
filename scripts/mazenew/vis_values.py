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

value_experiment = utils.load_diffusion(
    args.loadbase, args.dataset, args.value_loadpath,
    epoch=args.value_epoch, seed=args.seed,
)


## initialize value guide
value_function = value_experiment.ema
guide_config = utils.Config(args.guide, model=value_function, verbose=False)
guide = guide_config()



# tests

target = [1.0, 1.0]
goal = [*target, 0, 0, 0, 0]
import torch
goal = torch.tensor(goal, device=args.device)
guide.set_goal(goal)


# for t0 in reversed(range(32)):
t0 = 0
point_list = [[1.0, 2.0], [1.0, 3.0], [2.0, 3.0], [3.0, 3.0], [-1.0, -1.0], [10.0, 10.0]]

for pt in point_list:
    x = torch.tensor([[0.0, 0.0, *(pt),  0.0, 0.0]], device=args.device)
    x = x.repeat(1, 32, 1)

    cond = {}
    t = torch.tensor([t0], device=args.device)
    y, grad = guide.gradients(x, cond, t)

    # print("====================== t0", t0)
    print('+=============== pt', pt)
    print('y', y.detach().cpu().numpy())
    print('grad', grad[0, -1,:].detach().cpu().numpy())