import gym
import matplotlib.pyplot as plt

# import envs.d4rl_pointmaze

env = gym.make('maze2d-mMaze2-v0')
dataset = env.get_dataset()

observation  = dataset['observations']

# plot the first and second observation

plt.figure()
plt.scatter(observation[:,0], observation[ :,1])
plt.savefig('first_observation.png')
