import numpy as np
import os
import json

import seaborn as sns
import matplotlib.pyplot as plt

if __name__ == "__main__":
    
    env_list = ['maze2d-umaze-v1', 'maze2d-medium-v1', 'maze2d-large-v1']
    task_list = ['single', 'multi']
    env = env_list[0]
    task = task_list[0]
    
    base_dir = f'/home/zihao/cehao/github_space/diffuser-HRL/logs/{env}/plans/release_H128_T64_LimitsNormalizer_b1_condFalse_D16'
    
    file_suffix = task+'_'
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

    print('mean:', np.mean(score_list))
    print('std:', np.std(score_list))

    mean = np.round(np.mean(score_list) * 100, 2)
    std = np.round(np.std(score_list) * 100, 2)
    
    plt.figure()
    sns.kdeplot(score_list, fill=True)
    plt.title(f'KDE Plot: {env} {task} ' + str(mean) + '±' + str(std))
    plt.xlabel('Value')
    plt.ylabel('Density')

    # Display the plot
    plt.savefig(f'./{env}_{task}.png', dpi=300)
    plt.show()