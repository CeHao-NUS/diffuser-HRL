import logging
from envs.d4rl_pointmaze import waypoint_controller
from envs.d4rl_pointmaze import maze_model, maze_layouts
from envs.d4rl_pointmaze.maze_strings import maze_name_space

import numpy as np
import pickle
import gzip
import h5py
import argparse
import os
import tqdm
import time
import skvideo.io

def reset_data():
    return {
        'observations': [],
        'actions': [],
        'images': [],
        'terminals': [],
        'timeouts': [],
        'rewards': [],
        'infos/goal': [],
        'infos/qpos': [],
        'infos/qvel': [],
    }

def append_data(data, s, a, img, tgt, done, timeout, reward, env_data):
    data['observations'].append(s)
    data['actions'].append(a)
    if img is not None:
        data['images'].append(img)
    data['terminals'].append(done)
    data['timeouts'].append(timeout)
    data['rewards'].append(reward)  # Add rewards to dataset
    data['infos/goal'].append(tgt)
    data['infos/qpos'].append(env_data.qpos.ravel().copy())
    data['infos/qvel'].append(env_data.qvel.ravel().copy())

def npify(data):
    for k in data:
        if k == 'terminals' or k == 'timeouts':
            dtype = np.bool_
        else:
            dtype = np.float32
        data[k] = np.array(data[k], dtype=dtype)

def sample_env_and_controller(args):
    if args.manual_maze:
        layout_str = maze_name_space[args.manual_maze]
    else:
        layout_str = maze_layouts.rand_layout(seed=args.seed, size=args.fixed_maze_size)
    env = maze_model.MazeEnv(layout_str, agent_centric_view=args.agent_centric)
    controller = waypoint_controller.WaypointController(layout_str)
    return env, controller

def reset_env(env, agent_centric=False):
    s = env.reset()
    env.set_target()
    
    if agent_centric:
        [env.render(mode='rgb_array') for _ in range(100)]    # so that camera can catch up with agent
    return s

def save_video(file_name, frames, fps=20, video_format='mp4'):
    print(f'Saving video to {file_name}')
    skvideo.io.vwrite(
        file_name,
        frames,
        inputdict={'-r': str(int(fps))},
        outputdict={'-f': video_format, '-pix_fmt': 'yuv420p'}
    )
    print(f'Video saved as {file_name}')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--render', action='store_true', help='Render trajectories')
    parser.add_argument('--noisy', action='store_true', help='Noisy actions')
    parser.add_argument('--agent_centric', action='store_true', help='Whether agent-centric images are rendered.')
    parser.add_argument('--save_images', action='store_true', help='Save rendered images')
    parser.add_argument('--save_video', action='store_true', help='Save video of the trajectories')
    parser.add_argument('--data_dir', type=str, default='.', help='Base directory for dataset')
    parser.add_argument('--file_name', type=str, default='dataset.hdf5', help='Name of the output HDF5 file')
    parser.add_argument('--num_samples', type=int, default=int(2e5), help='Num samples to collect')
    parser.add_argument('--min_traj_len', type=int, default=int(50), help='Min number of samples per trajectory')
    parser.add_argument('--fixed_maze_size', type=int, default=int(40), help='Size of generated maze')
    parser.add_argument('--seed', type=int, default=None, help='Seed index')
    parser.add_argument('--manual_maze', default=None, help='Manually set maze layout')
    args = parser.parse_args()

    # Expand the `~` symbol in data_dir to the full path
    args.data_dir = os.path.expanduser(args.data_dir)

    max_episode_steps = 1600
    env, controller = sample_env_and_controller(args)

    # Initialize a dictionary to accumulate data across rollouts
    all_data = {
        'observations': [],
        'actions': [],
        'rewards': [],
        'terminals': [],
        'timeouts': [],
        'infos/goal': [],
        'infos/qpos': [],
        'infos/qvel': [],
        'images': []  # Store images if --save_images is enabled
    }

    all_video_frames = []  # Store video frames if --save_video is enabled

    s = reset_env(env, agent_centric=args.agent_centric)
    data = reset_data()
    ts = 0
    goal = np.array(env._target)  # Target (goal) position
    threshold = 0.5  # Distance threshold to consider the agent at the goal

    for tt in tqdm.tqdm(range(args.num_samples)):
        position = s[0:2]
        velocity = s[2:4]

        try:
            act, done = controller.get_action(position, velocity, env._target)
        except ValueError:
            # Failed to find valid path to goal
            data = reset_data()
            env, controller = sample_env_and_controller(args)
            s = reset_env(env, agent_centric=args.agent_centric)
            ts = 0
            continue

        if args.noisy:
            act = act + np.random.randn(*act.shape) * 0.5

        act = np.clip(act, -1.0, 1.0)
        if ts >= max_episode_steps:
            done = True

        reward = -np.linalg.norm(env._target - position)  # Simple reward based on distance to goal

        # Capture images if --save_images or --save_video is enabled
        img = None
        if args.save_images or args.save_video:
            img = env.render(mode='rgb_array')
            if args.save_images:
                all_data['images'].append(img)
            if args.save_video:
                all_video_frames.append(img)

        # Check if agent is within the goal threshold
        at_goal = np.linalg.norm(position - goal) < threshold

        # Add timeouts based on whether the agent leaves the goal after being there
        if len(data['observations']) > 0:
            previous_position = data['observations'][-1][:2]
            prev_at_goal = np.linalg.norm(previous_position - goal) < threshold
            timeout = prev_at_goal and not at_goal  # Timeout if previously at goal, but now not
            # if timeout:
                # print(f"Timeout at step {ts}")
        else:
            timeout = False  # No timeout at the first step

        # Append to data for this rollout
        append_data(data, s, act, img, env._target, done, timeout, reward, env.sim.data)

        ns, _, _, _ = env.step(act)
        ts += 1

        if done:
            # print(f"Episode done at step {tt}")
            if len(data['actions']) > args.min_traj_len:
                # Concatenate the data of this rollout to the full dataset
                all_data['observations'].extend(data['observations'])
                all_data['actions'].extend(data['actions'])
                all_data['rewards'].extend(data['rewards'])
                all_data['terminals'].extend(data['terminals'])
                all_data['timeouts'].extend(data['timeouts'])
                all_data['infos/goal'].extend(data['infos/goal'])
                all_data['infos/qpos'].extend(data['infos/qpos'])
                all_data['infos/qvel'].extend(data['infos/qvel'])

            data = reset_data()
            env, controller = sample_env_and_controller(args)
            s = reset_env(env, agent_centric=args.agent_centric)
            ts = 0
        else:
            s = ns

        if args.render:
            env.render(mode='human')

    # After all samples, save the concatenated dataset
    save_data(args, all_data)

    # Save video if required
    if args.save_video:
        video_file = args.file_name.replace('.hdf5', '.mp4')
        save_video(os.path.join(args.data_dir, video_file), all_video_frames)

def save_data(args, all_data):
    dir_name = ''
    os.makedirs(os.path.join(args.data_dir, dir_name), exist_ok=True)
    file_name = os.path.join(args.data_dir, dir_name, args.file_name)

    # Save the full concatenated dataset into a single HDF5 file
    with h5py.File(file_name, "w") as f:
        # Save the core data arrays from all_data dictionary
        for key, data in all_data.items():
            if key == 'images':  
                if args.save_images:
                    f.create_dataset(key, data=np.array(data, dtype=np.uint8))  # Save images as uint8
            else:
                f.create_dataset(key, data=np.array(data))


    print(f"Dataset saved at {file_name}")

if __name__ == "__main__":
    main()
