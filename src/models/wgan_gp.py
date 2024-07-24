import tensorflow as tf
import numpy as np

def create_generator(input_shape):
    '''
    Creates the generator model for the WGAN-GP.

    This function builds a generator neural network that takes a latent vector
    as input and generates an image. The architecture uses transposed convolutions
    to progressively upsample the input to the desired image size.

    Parameters:
        input_shape (int): The size of the input latent vector.

    Returns:
        tf.keras.Model: A Keras model representing the generator.

    Architecture:
        1. Dense layer to reshape the input
        2. Multiple transposed convolution layers with batch normalization and LeakyReLU
        3. Final transposed convolution layer with tanh activation

    Note:
        The generator produces images with 3 color channels (RGB).
    '''
    input_z_layer = tf.keras.layers.Input(shape=(input_shape, ))
    
    z = tf.keras.layers.Dense(4*4*256, use_bias=False)(input_z_layer)
    z = tf.keras.layers.Reshape((4, 4, 256))(z)
    
    x = tf.keras.layers.Conv2DTranspose(256, kernel_size=(4, 4), strides=(1, 1), padding='same', use_bias=False, kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02))(z)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)
    
    x = tf.keras.layers.Conv2DTranspose(128, kernel_size=(4, 4), strides=(2, 2), padding='same', use_bias=False, kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)
    
    x = tf.keras.layers.Conv2DTranspose(64, kernel_size=(4, 4), strides=(2, 2), padding='same', use_bias=False, kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)
    
    x = tf.keras.layers.Conv2DTranspose(32, kernel_size=(4, 4), strides=(2, 2), padding='same', use_bias=False, kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)
    
    output = tf.keras.layers.Conv2DTranspose(3, (4, 4), strides=(2, 2), padding='same', use_bias=False, activation="tanh",
                             kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02))(x)
    
    model = tf.keras.Model(inputs=input_z_layer, outputs=output, name='generator')

    return model

def create_discriminator(input_shape):
    '''
    Creates the discriminator model for the WGAN-GP.

    This function builds a discriminator neural network that takes an image
    as input and outputs a single value. The architecture uses convolutional
    layers to progressively downsample the input image.

    Parameters:
        input_shape (tuple): The shape of the input image (height, width, channels).

    Returns:
        tf.keras.Model: A Keras model representing the discriminator.

    Architecture:
        1. Multiple convolutional layers with layer normalization and LeakyReLU
        2. Flatten layer
        3. Dense layer for final output

    Note:
        The discriminator does not use a final activation function, as is typical
        in Wasserstein GANs.
    '''
    input_x_layer = tf.keras.layers.Input(shape=input_shape)
    
    x = tf.keras.layers.Conv2D(filters=32, kernel_size=(4, 4), strides=(2, 2), padding='same', use_bias=False, kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02))(input_x_layer)
    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)
    
    x = tf.keras.layers.Conv2D(filters=64, kernel_size=(4, 4), strides=(2, 2), padding='same', use_bias=False, kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02))(x)
    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)
    
    x = tf.keras.layers.Conv2D(filters=128, kernel_size=(4, 4), strides=(2, 2), padding='same', use_bias=False, kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02))(x)
    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)
    
    x = tf.keras.layers.Conv2D(filters=256, kernel_size=(4, 4), strides=(2, 2), padding='same', use_bias=False, kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02))(x)
    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)
    
    x = tf.keras.layers.Conv2D(1, kernel_size=(4, 4), strides=(1, 1), padding='same', use_bias=False, kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02))(x)
    
    x = tf.keras.layers.Flatten()(x)
    
    output = tf.keras.layers.Dense(units=1)(x)
    
    model = tf.keras.Model(inputs=input_x_layer, outputs=output, name='discriminator')

    return model

# WGAN Model
class WGAN_GP(tf.keras.Model):
    '''
    Wasserstein GAN with Gradient Penalty (WGAN-GP) implementation.

    This class implements the WGAN-GP architecture, which is an improved version
    of the original Wasserstein GAN. It uses gradient penalty to enforce the
    Lipschitz constraint on the discriminator.

    Attributes:
        generator (tf.keras.Model): The generator model.
        discriminator (tf.keras.Model): The discriminator model.
        latent_dim (int): The dimension of the latent space.
        gp_weight (float): The weight of the gradient penalty term.

    Methods:
        compile: Configures the model for training.
        gradient_penalty: Calculates the gradient penalty.
        train_step: Performs a single training step.
    '''

    def __init__(self, generator, discriminator, latent_dim, gp_weight:float=10.0, d_steps:int=5):
        '''
        Initializes the WGAN-GP model.

        Parameters:
            generator (tf.keras.Model): The generator model.
            discriminator (tf.keras.Model): The discriminator model.
            latent_dim (int): The dimension of the latent space.
            gp_weight (float): The weight of the gradient penalty term.
        '''
        super(WGAN_GP, self).__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.latent_dim = latent_dim
        self.gp_weight = gp_weight
        self.d_steps = d_steps

    def compile(self, disc_optimizer, gen_optimizer, disc_loss_fn, gen_loss_fn):
        '''
        Configures the model for training.

        Parameters:
            disc_optimizer: Optimizer for the discriminator.
            gen_optimizer: Optimizer for the generator.
            disc_loss_fn: Loss function for the discriminator.
            gen_loss_fn: Loss function for the generator.
        '''
        super(WGAN_GP, self).compile()
        self.disc_optimizer = disc_optimizer
        self.gen_optimizer = gen_optimizer
        self.disc_loss_fn = disc_loss_fn
        self.gen_loss_fn = gen_loss_fn

    @tf.function
    def gradient_penalty(self, batch_size, real_images, fake_images):
        '''
        Calculates the gradient penalty.

        This method enforces the Lipschitz constraint on the discriminator.

        Parameters:
            batch_size (int): The size of the batch.
            real_images (tf.Tensor): A batch of real images.
            fake_images (tf.Tensor): A batch of generated images.

        Returns:
            tf.Tensor: The calculated gradient penalty.
        '''
        alpha = tf.random.uniform([batch_size, 1, 1, 1], minval=0.0, maxval=1.0)
        diff = fake_images - real_images
        interpolated = real_images + alpha * diff

        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated)
            pred = self.discriminator(interpolated, training=True)

        grads = gp_tape.gradient(pred, [interpolated])[0]
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
        gp = tf.reduce_mean((norm - 1.0) ** 2)

        return gp

    @tf.function
    def train_step(self, real_images):
        '''
        Performs a single training step.

        This method trains both the generator and the discriminator.

        Parameters:
            real_images (tf.Tensor): A batch of real images.

        Returns:
            dict: A dictionary containing the discriminator and generator losses.
        '''
        batch_size = tf.shape(real_images)[0]

        for _ in range(self.d_steps):
            random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
            with tf.GradientTape() as tape:
                fake_images = self.generator(random_latent_vectors, training=True)
                fake_logits = self.discriminator(fake_images, training=True)
                real_logits = self.discriminator(real_images, training=True)

                d_cost = self.disc_loss_fn(real_logits, fake_logits)
                gp = self.gradient_penalty(batch_size, real_images, fake_images)
                disc_loss = d_cost + gp * self.gp_weight

            d_gradient = tape.gradient(disc_loss, self.discriminator.trainable_variables)
            self.disc_optimizer.apply_gradients(zip(d_gradient, self.discriminator.trainable_variables))

        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
        
        with tf.GradientTape() as tape:
            generated_images = self.generator(random_latent_vectors, training=True)
            gen_img_logits = self.discriminator(generated_images, training=True)
            gen_loss = self.gen_loss_fn(gen_img_logits)

        gen_gradient = tape.gradient(gen_loss, self.generator.trainable_variables)
        self.gen_optimizer.apply_gradients(zip(gen_gradient, self.generator.trainable_variables))

        return {"disc_loss": disc_loss, "gen_loss": gen_loss}
