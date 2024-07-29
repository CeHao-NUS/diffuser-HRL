import pdb

import diffuser.sampling as sampling
import diffuser.utils as utils
import numpy as np
from os.path import join
import json

import matplotlib.pyplot as plt
import seaborn as sns
#-----------------------------------------------------------------------------#
#----------------------------------- setup -----------------------------------#
#-----------------------------------------------------------------------------#

class Parser(utils.Parser):
    dataset: str = 'maze2d-umaze-v1'
    config: str = 'config.mazeguided.maze2d'

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


# ==================================================
def get_value_function(guide, target, x_range=[0, 4], y_range=[0, 4], t0=0, n_points=25):

    # target = [2.0, 3.0]
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

    x = np.linspace(x_range[0], x_range[1], n_points)
    y = np.linspace(y_range[0], y_range[1], n_points)
    xx, yy = np.meshgrid(x, y)

    xy_list = np.c_[xx.ravel(), yy.ravel()]

    values = []
    grads = []
    for pt in xy_list:

        observation = np.array([*pt, 0.0, 0.0], dtype=np.float32)
        action = np.array([0.0, 0.0], dtype=np.float32)
        obs_norm = dataset.normalizer.normalize(observation, 'observations')
        act_norm = dataset.normalizer.normalize(action, 'actions')

        obs_norm = torch.tensor(obs_norm, device=args.device)
        act_norm = torch.tensor(act_norm, device=args.device)

        x_input = torch.cat([act_norm, obs_norm], dim=0)

        x_input = x_input.repeat(1, 32, 1)

        cond = {}

        t = torch.tensor([t0], device=args.device)
        value, grad = guide.gradients(x_input, cond, t)
        # print('+=============== pt', pt)
        # print('y', y.detach().cpu().numpy())
        # print('grad', grad[0, -1,:].detach().cpu().numpy())

        value = value.detach().cpu().numpy()
        values.append(value)
        grad = grad[0, -1, 2:4].detach().cpu().numpy()
        grads.append(grad)

    values = np.array(values)
    values = values.reshape(n_points, n_points)
    grads = np.array(grads)
    grads = grads.reshape(n_points, n_points, 2)
    return values, grads, x, y


def plot_image(target = [2.0, 3.0], t0_list=[], image_prefix=''):
    values, grads, x, y = get_value_function(guide, target, x_range=[0, 4], y_range=[0, 4], n_points=25)


    # plot value
    plt.figure(figsize=(10, 8))
    sns.heatmap(values, xticklabels=np.round(x, 2), yticklabels=np.round(y, 2), cmap='viridis')

    # Find the indices for the target point
    target_index_x = np.argmin(np.abs(x - target[0]))
    target_index_y = np.argmin(np.abs(y - target[1]))
    # Plot the target point
    plt.scatter(
        target_index_x, 
        target_index_y, 
        color='red', 
        marker='*', 
        s=100
    )

    import os
    if not os.path.exists('images/viz'):
        os.makedirs('images/viz')

    plt.gca().invert_yaxis()
    plt.title('value map, target: {}'.format(target))
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig(f'images/viz/{target}_value_map.png', dpi=300)
    # plt.show()
    plt.close()


    # plot gradient
    plt.figure(figsize=(10, 10))
    plt.quiver(x, y, grads[:, :, 0], grads[:, :, 1], cmap='viridis')
    plt.scatter(target[0], target[1], color='red', marker='*', s=100)
    plt.title('gradient map, target: {}'.format(target))
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig(f'images/viz/{target}_gradient_map.png', dpi=300)
    # plt.show()
    plt.close()


target_list = [[1.0, 1.0], [1.0, 2.0], [1.0, 3.0], [2.0, 3.0], [3.0, 3.0], [3.0, 2.0], [3.0, 1.0]]

for target in target_list:
    # make them have t0 to iterate
    # name: target, t, 
    plot_image(target)