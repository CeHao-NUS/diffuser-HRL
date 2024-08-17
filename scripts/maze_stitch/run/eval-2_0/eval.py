import numpy as np
import os
import json

import seaborn as sns
import matplotlib.pyplot as plt

def plot_diffusion(subfolder, env):
    base_dir = f'./logs/{env}/plans/{subfolder}'
    
    file_suffix = 'eval_'
    rollout_name = 'rollout.json'

    score_list = []

    for idx in range(150):
        file_name = file_suffix  + str(idx)

        file_path = os.path.join(base_dir, file_name, rollout_name)
        
        # load json file
        with open(file_path, 'r') as f:
            data = json.load(f)
            score = data['score']   
            score_list.append(score)


    mean = np.round(np.mean(score_list) * 100, 2)
    std = np.round(np.std(score_list) * 100, 2)

    print(f'{env} {subfolder} ' + str(mean) + '±' + str(std))

    plt.figure()
    sns.kdeplot(score_list, fill=True)
    plt.title(f'KDE Plot: {env} {subfolder} ' + str(mean) + '±' + str(std))
    plt.xlabel('Value')
    plt.ylabel('Density')

    # Display the plot
    plt.savefig(f'./images/{env}_{subfolder}.png', dpi=300)
    # plt.show()
    plt.close()

if __name__ == '__main__':
    env_list = ['maze2d-umaze-v1', 'maze2d-medium-v1', 'maze2d-large-v1']
    env = env_list[0]

    subfolder = 'diff_H128_T32_L3_condFalse'
    plot_diffusion(subfolder, env)