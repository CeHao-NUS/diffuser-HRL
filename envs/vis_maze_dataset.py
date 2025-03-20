# import gym
# import matplotlib.pyplot as plt

# import envs.d4rl_pointmaze
# import d4rl

# env = gym.make('maze2d-medium-v1')
# dataset = env.get_dataset()

# observation  = dataset['observations']

# # plot the first and second observation

# plt.figure()
# plt.scatter(observation[:,0], observation[ :,1])
# plt.axis('equal')
# plt.savefig('images/first_observation.png')
# print('save to images/first_observation.png')


import gym
import matplotlib.pyplot as plt
import envs.d4rl_pointmaze
import d4rl
import numpy as np
import os
from PIL import Image

# Initialize environment and get dataset
env = gym.make('maze2d-medium-v1')
dataset = env.get_dataset()
observations = dataset['observations']


# Parameters for segmenting and plotting
n = 2000
segments = np.array_split(observations, n)  # Split into 200 segments

# Create directory if it doesn't exist
output_dir = 'images/segments'
os.makedirs(output_dir, exist_ok=True)

# Plot each segment, save individually, and rotate
for i, segment in enumerate(segments):
    path_length = len(segment)  # Determine the length of each path segment
    colors = plt.cm.jet(np.linspace(0, 1, path_length))  # Create a color gradient for the path
    
    plt.figure()
    plt.scatter(segment[:, 0], segment[:, 1], c=colors, alpha=0.7)
    plt.axis('equal')
    plt.axis('off')  # Turn off axes for clarity
    plt.colorbar(plt.cm.ScalarMappable(cmap='jet'), label='Path Progress')

    # Save each plot as an individual image
    filepath = os.path.join(output_dir, f'{i + 1}.png')
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()

    # Open and rotate the image
    with Image.open(filepath) as img:
        rotated_img = img.rotate(-90, expand=True)  # Rotate 90 degrees clockwise
        rotated_img.save(filepath)
    
    print(f'Saved rotated segment {i + 1} to {filepath}')

print('All segments saved and rotated in the "images/segments" folder.')


