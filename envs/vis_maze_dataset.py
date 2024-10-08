import gym
import matplotlib.pyplot as plt

import envs.d4rl_pointmaze

env = gym.make('maze2d-test-v0')
dataset = env.get_dataset()
print('length of dataset:', len(dataset))

observation  = dataset['observations']

# plot the first and second observation

plt.figure()
plt.scatter(observation[:,0], observation[ :,1])
plt.savefig('images/first_observation.png')
print('save to images/first_observation.png')
