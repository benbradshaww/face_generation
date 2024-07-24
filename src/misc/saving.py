import pandas as pd
import matplotlib.pyplot as plt

def generate_and_save_images(model, noise, epoch: int, image_dir: str):
    '''
    Generates and saves images using the provided GAN model.

    Parameters:
    model (tensorflow.keras.Model): The GAN model used for generating images.
    noise (numpy.ndarray): Input noise to the generator model.
    epoch (int): The current epoch number, used for naming saved images.
    image_dir (str): Directory path where the generated images will be saved.

    Returns:
    None
    '''
    predictions = model.generator(noise, training=False)

    _, axes = plt.subplots(4, 4, figsize=(8, 8))
    axes = axes.flatten()

    for i in range(predictions.shape[0]):
        axes[i].imshow((predictions[i] * 0.5) + 0.5)
        axes[i].axis('off')

    if epoch % 10 == 0:
        plt.savefig(image_dir + 'image_at_epoch_{:04d}.png'.format(int(epoch)))

    plt.tight_layout(pad=0.5)
    plt.show()

def save_loss(loss_path: str, gen_loss: float, disc_loss: float):
    '''
    Saves the generator and discriminator loss values to a CSV file.

    Parameters:
    loss_path (str): The file path to the CSV file where loss values will be saved.
    gen_loss (float): The generator loss value.
    disc_loss (float): The discriminator loss value.

    Returns:
    None
    '''
    try:
        df = pd.read_csv(loss_path)
    except FileNotFoundError:
        df = pd.DataFrame(columns=['gen_loss', 'disc_loss'])

    new_row = {'gen_loss': gen_loss, 'disc_loss': disc_loss}
    df.loc[len(df)] = new_row

    df.to_csv(loss_path, index=False)
