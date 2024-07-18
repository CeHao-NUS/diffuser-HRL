import json
import numpy as np
from os.path import join
import pdb

from diffuser.guides.policies import Policy
import diffuser.datasets as datasets
import diffuser.utils as utils


class Parser(utils.Parser):
    dataset: str = 'maze2d-umaze-v1'
    config: str = 'config.hier.maze2d_hl'


class ll_Parser(utils.Parser):
    dataset: str = 'maze2d-umaze-v1'
    config: str = 'config.hier.maze2d_ll'

#---------------------------------- setup ----------------------------------#

args = Parser().parse_args('plan')

# logger = utils.Logger(args)

env = datasets.load_environment(args.dataset)

#---------------------------------- loading ----------------------------------#

diffusion_experiment = utils.load_diffusion(args.logbase, args.dataset, args.diffusion_loadpath, epoch=args.diffusion_epoch)

diffusion = diffusion_experiment.ema
dataset = diffusion_experiment.dataset
renderer = diffusion_experiment.renderer

policy = Policy(diffusion, dataset.normalizer)


# ========================== set LL ==========================
# Parser().config = 'config.hier.maze2d_ll'
ll_args = ll_Parser().parse_args('plan')
ll_diffusion_experiment = utils.load_diffusion(ll_args.logbase, ll_args.dataset, ll_args.diffusion_loadpath, epoch=ll_args.diffusion_epoch)

ll_diffusion = ll_diffusion_experiment.ema
ll_dataset = ll_diffusion_experiment.dataset
ll_renderer = ll_diffusion_experiment.renderer

ll_policy = Policy(ll_diffusion, ll_dataset.normalizer)


#---------------------------------- main loop ----------------------------------#

observation = env.reset()

if args.conditional:    
    if args.init_pose is not None:
        observation = env.reset_to_location(args.init_pose)
        print('Resetting agent init to: ', observation)

    if args.target is not None:
        env.set_target(args.target)
        print('Resetting target to: ', env.get_target())
    

## set conditioning xy position to be the goal
target = env._target
# cond = {
#     diffusion.horizon - 1: np.array([*target, 0, 0]),
# }

cond = {
    args.goal_length - 1: np.array([*target, 0, 0]),
}


# ============== running ==============

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
        hl_sequence = samples.observations[0]

        sequence = np.array([])
        # ============== LL ==============
        goal_length = args.goal_length

        for idx in range(goal_length-1):
            ll_cond = {
                0: hl_sequence[idx],
                ll_diffusion.horizon - 1: hl_sequence[idx+1],
            }
            ll_action, ll_samples = ll_policy(ll_cond, batch_size=args.batch_size)
            ll_sequence = ll_samples.observations[0][:-1]

            if sequence.size == 0:
                sequence = ll_sequence 
            else:
                sequence = np.concatenate([sequence, ll_sequence], axis=0)

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
    print(
        f't: {t} | r: {reward:.2f} |  R: {total_reward:.2f} | score: {score:.4f} | '
        f'{action}'
    )

    if 'maze2d' in args.dataset:
        xy = next_observation[:2]
        goal = env.unwrapped._target
        print(
            f'maze | pos: {xy} | goal: {goal}'
        )

    ## update rollout observations
    rollout.append(next_observation.copy())

    # logger.log(score=score, step=t)

    hl_fullpath = join(args.savepath, 'HL.png')
    # if t == 0: renderer.composite(hl_fullpath, samples.observations, ncol=1)
    if t == 0: renderer.composite(hl_fullpath, samples.observations[:, : args.goal_length], ncol=1,
                                  conditions=cond)

    whole_path = join(args.savepath, 'FULL.png')
    if t == 0: renderer.composite(whole_path, np.array([sequence]), ncol=1,
                                  conditions=cond)
    
    # if t % args.vis_freq == 0 or terminal:
    if terminal or t == env.max_episode_steps-1:

        # renderer.render_plan(join(args.savepath, f'{t}_plan.mp4'), samples.actions, samples.observations, state)

        ## save rollout thus far
        renderer.composite(join(args.savepath, 'rollout.png'), np.array(rollout)[None], ncol=1,
                           conditions=cond)

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
