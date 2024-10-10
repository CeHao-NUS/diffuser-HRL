import numpy as np
import os
import json

import seaborn as sns
import matplotlib.pyplot as plt

from PIL import Image, ImageDraw, ImageFont


def plot_diffusion(subfolder, env):
    base_dir = f'./logs/{env}/plans/{subfolder}'
    
    file_suffix = 'eval_'
    rollout_name = 'rollout.json'
    photo_name = 'whole.png'

    score_list = []
    image_list = []
    failed_list = {}

    for idx in range(150):
        file_name = file_suffix  + str(idx)
        file_path = os.path.join(base_dir, file_name, rollout_name)
        photo_path = os.path.join(base_dir, file_name, photo_name)

        image_list.append(photo_path)
        
        # load json file
        with open(file_path, 'r') as f:
            data = json.load(f)
            score = data['score']   
            score_list.append(score)
            if score < 0.1:
                failed_list[idx] = score


    mean = np.round(np.mean(score_list) * 100, 2)
    std = np.round(np.std(score_list) * 100, 2)

    print(f'{env} {subfolder} ' + str(mean) + '±' + str(std))

    plt.figure()
    sns.kdeplot(score_list, fill=True)
    plt.title(f'KDE Plot: {env} {subfolder} ' + str(mean) + '±' + str(std))
    plt.xlabel('Value')
    plt.ylabel('Density')

    # create a folder to save the images
    if not os.path.exists('./images'):
        os.makedirs('./images')

    # Display the plot
    plt.savefig(f'./images/{env}_{subfolder}.png', dpi=300)
    # plt.show()
    plt.close()
    print(f'KDE Plot saved to ./images/{env}_{subfolder}.png')


    
    # save the content  failed list to './failed_list.txt'
    failed_path = base_dir + '/failed_list.txt'

    with open(failed_path, 'w') as f:
        for key, value in failed_list.items():
            f.write(f'{key}: {value}\n')

    print(f'Failed list saved to {failed_path}')

    # Concatenate images
    image_names = [f'{i}' for i in range(150)]
    concatenate_image_name = base_dir +  f'/{env}_{subfolder}_concatenated.png'
    concatenate_images_with_custom_titles(image_list, concatenate_image_name, image_names, max_columns=5, font_size=100)

from PIL import Image, ImageDraw, ImageFont
import os

def concatenate_images_with_custom_titles(image_paths, output_path, names, max_columns=10, font_size=40):
    # Load images and calculate max width and height
    images = [Image.open(image) for image in image_paths]

    # Font settings with a larger font size
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", font_size)
    except IOError:
        font = ImageFont.load_default()
        print("Font not found, using default font")
    
    # Calculate title height using getbbox
    title_height = max([font.getbbox(name)[3] for name in names])

    max_width = max(image.width for image in images)
    max_height = max(image.height for image in images) + title_height

    # Calculate the grid size
    total_images = len(images)
    rows = (total_images + max_columns - 1) // max_columns
    grid_width = max_width * min(max_columns, total_images)
    grid_height = max_height * rows

    # Create a new blank image with the calculated size
    new_image = Image.new('RGB', (grid_width, grid_height), (255, 255, 255))
    draw = ImageDraw.Draw(new_image)

    # Paste each image into the grid
    for index, (image, name) in enumerate(zip(images, names)):
        col = index % max_columns
        row = index // max_columns
        x_offset = col * max_width
        y_offset = row * max_height

        # Draw the name with the specified font size
        name_width, _ = draw.textbbox((0, 0), name, font=font)[2:]
        name_x = x_offset + (max_width - name_width) // 2
        draw.text((name_x, y_offset), name, fill="black", font=font)

        # Paste the image below the name
        new_image.paste(image, (x_offset, y_offset + title_height))

    # Save the resulting image
    new_image.save(output_path)
    print(f"Concatenated image saved as {output_path}")



if __name__ == '__main__':
    env_list = ['maze2d-umaze-v1', 'maze2d-medium-v1', 'maze2d-large-v1', 'maze2d-testbig-v0']
    env = env_list[1]

    subfolder = 'single_coupled_forwardnoise_H256_T256_d0.99_b1_condFalse'
    plot_diffusion(subfolder, env)