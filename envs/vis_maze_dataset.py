import gym
import matplotlib.pyplot as plt

import envs.d4rl_pointmaze
import d4rl

env = gym.make('maze2d-large-v1')
dataset = env.get_dataset()

observation  = dataset['observations']

# plot the first and second observation

plt.figure()
plt.scatter(observation[:,0], observation[ :,1])
plt.axis('equal')
plt.savefig('images/first_observation.png')
print('save to images/first_observation.png')
