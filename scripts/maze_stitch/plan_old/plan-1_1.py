
# plan for LL joint stitch

import json
import numpy as np
from os.path import join
import pdb

from diffuser.guides.policies import Policy
import diffuser.datasets as datasets
import diffuser.utils as utils



class Parser(utils.Parser):
    dataset: str = 'maze2d-umaze-v1'
    config: str = 'config.stitch.plan.plan_1_1'

def stitch_batches(x):
    # flatten, x.shape = (batch, time, dim)
    x = x.reshape(-1, x.shape[-1])

    # create a dim at 0
    # x = x.unsqueeze(0)
    x = x[np.newaxis, :]
    return x

#---------------------------------- setup ----------------------------------#

args = Parser().parse_args('plan')

# logger = utils.Logger(args)

env = datasets.load_environment(args.dataset)

#---------------------------------- loading ----------------------------------#



diffusion_experiment = utils.load_diffusion(
    args.loadbase, args.dataset, args.diffusion_loadpath,
    epoch=args.diffusion_epoch, seed=args.seed,
    device=args.device
)

diffusion = diffusion_experiment.ema
dataset = diffusion_experiment.dataset
renderer = diffusion_experiment.renderer

# policy = Policy(diffusion, dataset.normalizer)

policy_config = utils.Config(
    args.policy,
    guide=None,
    scale=args.scale,
    diffusion_model=diffusion,
    normalizer=dataset.normalizer,
    preprocess_fns=args.preprocess_fns,
    ## sampling kwargs
    sample_fn=args.sample_fun,
    n_guide_steps=args.n_guide_steps,
    t_stopgrad=args.t_stopgrad,
    scale_grad_by_std=args.scale_grad_by_std,
    verbose=False,

    batch_size=args.seg_length,
)

policy = policy_config()

#---------------------------------- main loop ----------------------------------#

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
    (args.seg_length-1, diffusion.horizon - 1): np.array([*target, 0, 0]),
}

cond_plot = {diffusion.horizon * args.seg_length - 1: np.array([*target, 0, 0])}

## observations for rendering
rollout = [observation.copy()]

total_reward = 0
for t in range(env.max_episode_steps):

    state = env.state_vector().copy()

    ## can replan if desired, but the open-loop plans are good enough for maze2d
    ## that we really only need to plan once
    if t == 0:
        cond[(0,0)] = observation
        # cond_plot[0] = observation

        action, samples = policy(cond, batch_size=args.batch_size)

        for idx in range(args.seg_length):
            cond_plot[diffusion.horizon * idx] = samples.observations[idx][0]

        actions_plan = stitch_batches(samples.actions)
        observation_plan = stitch_batches(samples.observations)

        actions = actions_plan[0]
        sequence = observation_plan[0]

        # actions = samples.actions[0]
        # sequence = samples.observations[0]
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
    # pdb.set_trace()
    ####

    # else:
    #     actions = actions[1:]
    #     if len(actions) > 1:
    #         action = actions[0]
    #     else:
    #         # action = np.zeros(2)
    #         action = -state[2:]
    #         pdb.set_trace()



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
        ll_fullpath = join(args.savepath, 'LL.png')
        renderer.composite(ll_fullpath, samples.observations, ncol=1,  conditions=cond)

        fullpath = join(args.savepath, 'whole.png')
        renderer.composite(fullpath, observation_plan, ncol=1, conditions=cond_plot)
    # renderer.render_plan(join(args.savepath, f'{t}_plan.mp4'), samples.actions, samples.observations, state)

    if terminal or t == env.max_episode_steps-1:
        ## save rollout thus far
        renderer.composite(join(args.savepath, 'rollout.png'), np.array(rollout)[None], ncol=1, conditions=cond_plot)

        # renderer.render_rollout(join(args.savepath, f'rollout.mp4'), rollout, fps=80)

        # logger.video(rollout=join(args.savepath, f'rollout.mp4'), plan=join(args.savepath, f'{t}_plan.mp4'), step=t)

    if terminal:
        break

    observation = next_observation

# logger.finish(t, env.max_episode_steps, score=score, value=0)

## save result as a json file
json_path = join(args.savepath, 'rollout.json')
json_data = {'score': score, 'step': t, 'return': total_reward, 'term': terminal,
    'epoch_diffusion': diffusion_experiment.epoch}
json.dump(json_data, open(json_path, 'w'), indent=2, sort_keys=True)
