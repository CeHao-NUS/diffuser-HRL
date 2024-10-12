
import pdb

import diffuser.sampling as sampling
import diffuser.utils as utils
import numpy as np
from os.path import join
import json

import envs.d4rl_pointmaze # to enable offline maze env
#-----------------------------------------------------------------------------#
#----------------------------------- setup -----------------------------------#
#-----------------------------------------------------------------------------#

class Parser(utils.Parser):
    dataset: str = 'maze2d-umaze-v1'
    config: str = 'config.stitch.plan.plan_1_0'

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


if args.value_loadpath is not None:

    value_experiment = utils.load_diffusion(
        args.loadbase, args.dataset, args.value_loadpath,
        epoch=args.value_epoch, seed=args.seed,
        device=args.device
    )

    ## ensure that the diffusion model and value function are compatible with each other
    utils.check_compatibility(diffusion_experiment, value_experiment)

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
    from diffuser.guides.policies import Policy
    policy = Policy(diffusion, dataset.normalizer)

#---------------------------------- main loop ----------------------------------#
env = dataset.env

observation = env.reset()

if args.conditional:    
    if args.init_pose is not None:
        observation = env.reset_to_location(args.init_pose)
        print('Resetting agent init to: ', observation)

    if args.target is not None:
        env.set_target(args.target)
        print('Resetting target to: ', env.get_target())
    else:
        env.set_target()
    

## set conditioning xy position to be the goal
target = env._target

# diffusion.horizon = args.plan_horizon

cond = {
    diffusion.horizon - 1: np.array([*target, 0, 0]),
}

print('horizon is', diffusion.horizon)

## observations for rendering
rollout = [observation.copy()]

total_reward = 0
for t in range(env.max_episode_steps):

    state = env.state_vector().copy()

    ## can replan if desired, but the open-loop plans are good enough for maze2d
    ## that we really only need to plan once
    if t == 0:
        cond[0] = observation

        action, samples = policy(cond, batch_size=args.batch_size)
        actions = samples.actions[0]
        sequence = samples.observations[0]
        print('!!!! last state', sequence[-1][:2], 'target', target[:2],
              'dist', np.linalg.norm(sequence[-1][:2] - target[:2]))
    # pdb.set_trace()

    # ####
    if t < len(sequence) - 1:
        next_waypoint = sequence[t+1]
    else:
        next_waypoint = sequence[-1].copy()
        next_waypoint[2:] = 0
        # pdb.set_trace()

    ## can use actions or define a simple controller based on state predictions
    action = next_waypoint[:2] - state[:2] + (next_waypoint[2:] - state[2:])

    next_observation, reward, terminal, _ = env.step(action)
    total_reward += reward
    score = env.get_normalized_score(total_reward)
    # print(
    #     f't: {t} | r: {reward:.2f} |  R: {total_reward:.2f} | score: {score:.4f} | '
    #     f'{action}'
    # )

    if 'maze2d' in args.dataset:
        xy = next_observation[:2]
        goal = env.unwrapped._target
        # print(
        #     f'maze | pos: {xy} | goal: {goal}'
        # )

    ## update rollout observations
    rollout.append(next_observation.copy())

    # if t % args.vis_freq == 0 or terminal:
    
    if t == 0:
        fullpath = join(args.savepath, 'LL.png')
        renderer.composite(fullpath, samples.observations[:, :, :diffusion.horizon], ncol=1, conditions=cond)

    if terminal or t == env.max_episode_steps-1:
        ## save rollout thus far
        renderer.composite(join(args.savepath, 'rollout.png'), np.array(rollout)[None], ncol=1,        
                           conditions=cond)

    if terminal:
        break

    observation = next_observation

import os
# ==================== save intermediate reconstructions
x_recon_store = policy.process_raw_trajectory()
for key, x_recon in x_recon_store.items():
    path_dir = join(args.savepath, 'x_recon', f'{key}.png')
    if not os.path.exists(join(args.savepath, 'x_recon')):
        os.makedirs(join(args.savepath, 'x_recon'))
    renderer.composite(path_dir, x_recon[:, :, :diffusion.horizon], ncol=1, conditions=cond)

# ==================== check forward and backward
x_bf_store, xt_store = policy.get_for_and_back()
for key, x_bf in x_bf_store.items():
    path_dir = join(args.savepath, 'x_bf', f'{key}.png')
    if not os.path.exists(join(args.savepath, 'x_bf')):
        os.makedirs(join(args.savepath, 'x_bf'))
    renderer.composite(path_dir, x_bf[:, :, :diffusion.horizon], ncol=1, conditions=cond)

for key, xt in xt_store.items():
    path_dir = join(args.savepath, 'x_bf', f'{key}_xt.png')
    renderer.composite(path_dir, xt[:, :, :diffusion.horizon], ncol=1, conditions=cond)

# ==================== sample again
sample2 = policy.sample_again()
path_dir = join(args.savepath, 'sample_again.png')
renderer.composite(path_dir, sample2.observations[:, :, :diffusion.horizon], ncol=1, conditions=cond)


# ==================== save intermediate reconstructions
x_recon_store = policy.process_raw_trajectory()
for key, x_recon in x_recon_store.items():
    path_dir = join(args.savepath, 'x_recon2', f'{key}.png')
    if not os.path.exists(join(args.savepath, 'x_recon2')):
        os.makedirs(join(args.savepath, 'x_recon2'))
    renderer.composite(path_dir, x_recon[:, :, :diffusion.horizon], ncol=1, conditions=cond)

# logger.finish(t, env.max_episode_steps, score=score, value=0)

## save result as a json file
json_path = join(args.savepath, 'rollout.json')
json_data = {'score': score, 'step': t, 'return': total_reward, 'term': terminal,
    'epoch_diffusion': diffusion_experiment.epoch}
json.dump(json_data, open(json_path, 'w'), indent=2, sort_keys=True)

