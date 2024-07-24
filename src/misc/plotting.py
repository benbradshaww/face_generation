import pandas as pd
import matplotlib.pyplot as plt
import imageio
import os
from IPython.display import Image

def plot_loss(loss_path: str):
    '''
    Plots the generator and discriminator loss over epochs.

    Parameters:
    loss_path (str): The file path to the CSV file containing loss values.

    Returns:
    None
    '''
    df = pd.read_csv(loss_path)
    
    plt.plot(df.index, df['gen_loss'], label='gen loss', color='blue')
    plt.plot(df.index, df['disc_loss'], label='disc loss', color='red')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


def create_gif_from_folder(folder_path, output_filename, fps=2):
    """
    Creates a GIF from images.

    Parameters:
    - folder_path (str): path to the folder containing the images
    - output_filename (str): name of the output GIF file
    - fps (int): frames per second for the GIF

    Returns:
    - Displays the GIF
    """
    images = []
    for file_name in sorted(os.listdir(folder_path)):
        if file_name.endswith(('png', 'jpg', 'jpeg', 'bmp', 'gif')):
            file_path = os.path.join(folder_path, file_name)
            images.append(imageio.imread(file_path))

    imageio.mimsave(output_filename, images, fps=fps)
    
    return Image(filename=output_filename)

