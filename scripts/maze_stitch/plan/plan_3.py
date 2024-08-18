'''
p-3_0: HL_fix + LL_fix
p-3_1: HL_varh(guide) + LL_fix
p-3_2: HL_fix + LL_varh(guide)
p-3_3: HL_varh(guide) + LL_varh(guide)

'''

import json
import numpy as np
from os.path import join
import pdb

from diffuser.guides.policies import Policy
import diffuser.datasets as datasets
import diffuser.utils as utils
import diffuser.sampling as sampling


class Parser(utils.Parser):
    dataset: str = 'maze2d-umaze-v1'
    config: str = 'config.stitch.plan.plan_2_0'

def stitch_batches(x):
    # flatten, x.shape = (batch, time, dim)
    x = x.reshape(-1, x.shape[-1])

    # create a dim at 0
    # x = x.unsqueeze(0)
    x = x[np.newaxis, :]
    return x

def check_subgoal_reached(observation, target, tol=0.05):
    return np.linalg.norm(observation[:2] - target[:2]) < tol

#---------------------------------- setup ----------------------------------#

args = Parser().parse_args('plan')

# logger = utils.Logger(args)

env = datasets.load_environment(args.dataset)

#---------------------------------- loading ----------------------------------#

# +++++++++++++ LL +++++++++++++ #

LL_diffusion_experiment = utils.load_diffusion(
    args.loadbase, args.dataset, args.LL_diffusion_loadpath,
    epoch=args.LL_diffusion_epoch, seed=args.seed,
)

LL_diffusion = LL_diffusion_experiment.ema
LL_dataset = LL_diffusion_experiment.dataset
LL_renderer = LL_diffusion_experiment.renderer

renderer = LL_renderer

if args.LL_value_loadpath is not None:

    LL_value_experiment = utils.load_diffusion(
        args.loadbase, args.dataset, args.LL_value_loadpath,
        epoch=args.LL_value_epoch, seed=args.seed,
    )

    ## ensure that the diffusion model and value function are compatible with each other
    utils.check_compatibility(LL_diffusion_experiment, LL_value_experiment)

    ## initialize value guide
    LL_value_function = LL_value_experiment.ema
    LL_guide_config = utils.Config(args.LL_guide, model=LL_value_function, verbose=False)
    LL_guide = LL_guide_config()

    ## policies are wrappers around an unconditional diffusion model and a value guide
    LL_policy_config = utils.Config(
        args.LL_policy,
        guide=LL_guide,
        scale=args.LL_scale,
        diffusion_model=LL_diffusion,
        normalizer=LL_dataset.normalizer,
        preprocess_fns=args.LL_preprocess_fns,
        ## sampling kwargs
        sample_fn=sampling.n_step_guided_p_sample,
        n_guide_steps=args.LL_n_guide_steps,
        t_stopgrad=args.LL_t_stopgrad,
        scale_grad_by_std=args.LL_scale_grad_by_std,
        verbose=False,
    )

    LL_policy = LL_policy_config()

else:
    LL_policy = Policy(LL_diffusion, LL_dataset.normalizer, batch_size=args.seg_length)


# +++++++++++++ HL +++++++++++++ #

HL_diffusion_experiment = utils.load_diffusion(
    args.loadbase, args.dataset, args.HL_diffusion_loadpath,
    epoch=args.HL_diffusion_epoch, seed=args.seed,
)

HL_diffusion = HL_diffusion_experiment.ema
HL_dataset = HL_diffusion_experiment.dataset
HL_renderer = HL_diffusion_experiment.renderer

# HL_policy = Policy(HL_diffusion, HL_dataset.normalizer)

horizon_HL =  args.HL_horizon // args.downsample 

if args.HL_value_loadpath is not None:
        
    HL_value_experiment = utils.load_diffusion(
        args.loadbase, args.dataset, args.HL_value_loadpath,
        epoch=args.HL_value_epoch, seed=args.seed,
    )

    ## ensure that the diffusion model and value function are compatible with each other
    utils.check_compatibility(HL_diffusion_experiment, HL_value_experiment)

    ## initialize value guide
    HL_value_function = HL_value_experiment.ema
    HL_guide_config = utils.Config(args.HL_guide, model=HL_value_function, verbose=False)
    HL_guide = HL_guide_config()

    ## policies are wrappers around an unconditional diffusion model and a value guide
    HL_policy_config = utils.Config(
        args.HL_policy,
        guide=HL_guide,
        scale=args.HL_scale,
        diffusion_model=HL_diffusion,
        normalizer=HL_dataset.normalizer,
        preprocess_fns=args.HL_preprocess_fns,
        ## sampling kwargs
        sample_fn=sampling.n_step_guided_p_sample,
        n_guide_steps=args.HL_n_guide_steps,
        t_stopgrad=args.HL_t_stopgrad,
        scale_grad_by_std=args.HL_scale_grad_by_std,
        verbose=False,
    )

    HL_policy = HL_policy_config()
else:
    HL_policy = Policy(HL_diffusion, HL_dataset.normalizer)

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

seg_length = args.seg_length
assert seg_length <= args.HL_horizon

HL_cond = {
    (0, seg_length): np.array([*target, 0, 0]),
}

cond_plot = {LL_diffusion.horizon * args.seg_length - 1: np.array([*target, 0, 0])}

## observations for rendering
rollout = [observation.copy()]

total_reward = 0
for t in range(env.max_episode_steps):

    state = env.state_vector().copy()

    ## can replan if desired, but the open-loop plans are good enough for maze2d
    ## that we really only need to plan once
    if t == 0:
        # +++++++++++++ HL +++++++++++++ #
        HL_cond[(0,0)] = observation
        HL_action, HL_samples = HL_policy(HL_cond, batch_size=args.batch_size)

        hl_obs = HL_samples.observations[0]

        for idx in range(args.seg_length):
            cond_plot[LL_diffusion.horizon * idx] = hl_obs[idx]

        # +++++++++++++ LL +++++++++++++ #
        LL_cond = {}
        for i in range(seg_length):
            LL_cond[(i,0)] = hl_obs[i]
            LL_cond[(i, args.LL_horizon - 1)] = hl_obs[i+1]

        LL_action, LL_samples = LL_policy(LL_cond, batch_size=args.batch_size)

        actions_plan = stitch_batches(LL_samples.actions)
        observation_plan = stitch_batches(LL_samples.observations)

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
        hl_fullpath = join(args.savepath, 'HL.png')
        renderer.composite(hl_fullpath, HL_samples.observations[:, :seg_length+1, :], ncol=1,  conditions=HL_cond)
        
        ll_fullpath = join(args.savepath, 'LL.png')
        renderer.composite(ll_fullpath, LL_samples.observations, ncol=1,  conditions=LL_cond)

        whole_fullpath = join(args.savepath, 'whole.png')
        renderer.composite(whole_fullpath, observation_plan, ncol=1,  conditions=cond_plot)
    
    # renderer.render_plan(join(args.savepath, f'{t}_plan.mp4'), samples.actions, samples.observations, state)

    if terminal or t == env.max_episode_steps-1:
        ## save rollout thus far
        renderer.composite(join(args.savepath, 'rollout.png'), np.array(rollout)[None], ncol=1,
                           conditions=HL_cond)

        # renderer.render_rollout(join(args.savepath, f'rollout.mp4'), rollout, fps=80)

        # logger.video(rollout=join(args.savepath, f'rollout.mp4'), plan=join(args.savepath, f'{t}_plan.mp4'), step=t)

    if terminal:
        break

    observation = next_observation

# logger.finish(t, env.max_episode_steps, score=score, value=0)

## save result as a json file
json_path = join(args.savepath, 'rollout.json')
json_data = {'score': score, 'step': t, 'return': total_reward, 'term': terminal,
    'epoch_diffusion': LL_diffusion_experiment.epoch}
json.dump(json_data, open(json_path, 'w'), indent=2, sort_keys=True)
