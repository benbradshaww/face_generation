import tensorflow as tf
from tensorflow import keras
import numpy as np

class MappingNetwork(keras.Model):
    def __init__(self, latent_dim, style_dim, num_layers):
<<<<<<< HEAD
=======
        '''
        Initializes the MappingNetwork model.

        Args:
            latent_dim (int): The dimensionality of the input latent vector z from a priori distribution.
            style_dim (int): The dimensionality of the output style vector.
            num_layers (int): The number of dense layers in the network.
        '''
>>>>>>> aa25840 (Laptop commit)
        super(MappingNetwork, self).__init__()
        self.network = keras.Sequential([
            keras.layers.Dense(style_dim, activation='relu')
            for _ in range(num_layers)
        ])
    
    def call(self, z):
<<<<<<< HEAD
=======
        '''
        Forward pass through the mapping network.

        Args:
            z (tf.Tensor): The input latent tensor z of shape (batch_size, latent_dim).

        Returns:
            tf.Tensor: The output style (tensor) of shape (batch_size, style_dim) after passing through the mapping network.
        '''
>>>>>>> aa25840 (Laptop commit)
        return self.network(z)

class AdaIN(keras.layers.Layer):
    def __init__(self):
<<<<<<< HEAD
        super(AdaIN, self).__init__()
    
    def call(self, x, style):
        mean = tf.reduce_mean(x, axis=[1, 2], keepdims=True)
        std = tf.math.reduce_std(x, axis=[1, 2], keepdims=True) + 1e-8
        y = (x - mean) / std
        
        style = tf.reshape(style, [-1, 1, 1, style.shape[-1]])
        return y * style[:, :, :, :x.shape[-1]] + style[:, :, :, x.shape[-1]:]

class StyleLayer(keras.layers.Layer):
    def __init__(self, filters, kernel_size, upsample=False):
=======
        '''
        Initialises the Adaptive Instance Normalization Layer
        '''
        super(AdaIN, self).__init__()

    def build(self, input_shape):
        self.channels = input_shape[-1]
        self.style_scale_transform = keras.layers.Dense(self.channels)
        self.style_shift_transform = keras.layers.Dense(self.channels)

    def call(self, x, style):
        '''
        Performs an adaptive instance normalization on input latent tensor 'x' using the style tensor.
        Args:
            x (tf.Tensor): The input latent tensor of shape (batch_size, height, width, channels)
            style (tf.Tensor): The input style tensor of shape (batch_size, style_dim)
        Returns:
        tf.Tensor: The output tensor after having applied adaptive instance normalized.
        '''
        # Instance normalization
        mean = tf.reduce_mean(x, axis=[1, 2], keepdims=True)
        std = tf.math.reduce_std(x, axis=[1, 2], keepdims=True) + 1e-8
        y = (x - mean) / std

        # Transform style
        style_scale = self.style_scale_transform(style)
        style_shift = self.style_shift_transform(style)

        # Reshape style parameters
        style_scale = tf.reshape(style_scale, [-1, 1, 1, self.channels])
        style_shift = tf.reshape(style_shift, [-1, 1, 1, self.channels])
        

        # Apply style
        return y * style_scale + style_shift

class StyleLayer(keras.layers.Layer):
    '''
    Initializes the StyleLayer. This layer includes an optional upsampling step,
    a convolution, Adaptive Instance Normalization (AdaIN), and an activation function.

    Args:
        filters (int): Number of filters for the convolutional layer.
        kernel_size (int): Size of the kernel for the convolutional layer.
        upsample (bool): Whether to include an upsampling step before the convolution. Default is False.
        add_noise (bool): Whether to add noise between the convolution and the adaptive instance normalization layer. Default is False.
    '''

    def __init__(self, filters, kernel_size, upsample=False, add_noise=False):
>>>>>>> aa25840 (Laptop commit)
        super(StyleLayer, self).__init__()
        self.upsample = upsample
        self.conv = keras.layers.Conv2D(filters, kernel_size, padding='same')
        self.adain = AdaIN()
        self.activation = keras.layers.LeakyReLU(0.2)
<<<<<<< HEAD
    
    def call(self, x, style):
=======
        self.add_noise = add_noise
        if add_noise:
            self.noise_scale = self.add_weight(
                name='noise_scale', 
                shape=(1, 1, 1, filters),
                initializer='zeros',
                trainable=True
            )
    
    def call(self, x, style):
        '''
        Applies the StyleLayer operations: optional upsampling, convolution, AdaIN, and activation.

        Args:
            x (tf.Tensor): The input tensor of shape (batch_size, height, width, channels).
            style (tf.Tensor): The style tensor of shape (batch_size, 2 * channels), containing 
                               the scale and shift parameters for AdaIN.

        Returns:
            tf.Tensor: The output tensor after applying the style layer operations.
        '''
        
>>>>>>> aa25840 (Laptop commit)
        if self.upsample:
            x = tf.image.resize(x, (x.shape[1]*2, x.shape[2]*2), method='bilinear')
        x = self.conv(x)
        x = self.adain(x, style)
<<<<<<< HEAD
=======

        if self.add_noise:
            noise = tf.random.normal(tf.shape(x), dtype=x.dtype)
            x += self.noise_scale * noise
        
>>>>>>> aa25840 (Laptop commit)
        return self.activation(x)

class Generator(keras.Model):
    def __init__(self, latent_dim, style_dim, num_layers, channels):
<<<<<<< HEAD
=======
        '''
        Initializes the Synthesis Network (Generator).

        Args:
            latent_dim (int): Dimensionality of the input latent vector.
            style_dim (int): Dimensionality of the style vector produced by the Mapping Network.
            num_layers (int): Number of layers in the Mapping Network.
            channels (list of int): List of channel sizes for each layer in the Synthesis Network.
        '''

>>>>>>> aa25840 (Laptop commit)
        super(Generator, self).__init__()
        self.mapping = MappingNetwork(latent_dim, style_dim, num_layers)
        self.initial_const = tf.Variable(tf.random.normal([1, 4, 4, channels[0]]))
        self.style_layers = [
<<<<<<< HEAD
            StyleLayer(ch, 3, upsample=True) for ch in channels[1:]
=======
            StyleLayer(filters=ch, kernel_size=3, upsample=True, add_noise=False) for ch in channels
>>>>>>> aa25840 (Laptop commit)
        ]
        self.to_rgb = keras.layers.Conv2D(3, 1, activation='tanh')
    
    def call(self, z):
<<<<<<< HEAD
        w = self.mapping(z)
        x = tf.tile(self.initial_const, [tf.shape(z)[0], 1, 1, 1])
=======
        '''
        Generates an image from the latent variable z.

        Args:
            z (tf.Tensor): A latent vector of shape (batch_size, latent_dim).

        Returns:
            tf.Tensor: Generated image of shape (batch_size, height, width, 3).
        '''

        w = self.mapping(z) # Generate Styles
        x = tf.tile(input=self.initial_const, multiples=[tf.shape(z)[0], 1, 1, 1])
>>>>>>> aa25840 (Laptop commit)
        
        for layer in self.style_layers:
            x = layer(x, w)
        
        return self.to_rgb(x)

class Discriminator(keras.Model):
    def __init__(self, channels):
<<<<<<< HEAD
        super(Discriminator, self).__init__()
        self.layers_list = [
            keras.layers.Conv2D(ch, 3, strides=2, padding='same', activation='leaky_relu')
=======
        '''
        Initializes the Discriminator network.

        Args:
            channels (list of int): List of channel sizes for each convolutional layer in the discriminator.
        '''
        super(Discriminator, self).__init__()
        self.layers_list = [
            keras.layers.Conv2D(filters=ch, kernel_size=3, strides=2, padding='same', activation=keras.layers.LeakyReLU(0.2))
>>>>>>> aa25840 (Laptop commit)
            for ch in reversed(channels)
        ]
        self.flatten = keras.layers.Flatten()
        self.fc = keras.layers.Dense(1)
<<<<<<< HEAD
    
    def call(self, x):
        for layer in self.layers_list:
            x = layer(x)
        x = self.flatten(x)
=======

    def call(self, x):
        '''
        Applies the Discriminator network to the input tensor.

        Args:
            x (tf.Tensor): The input tensor of shape (batch_size, height, width, channels).

        Returns:
            tf.Tensor: The output tensor of shape (batch_size, 1), representing the discriminator's confidence.
        '''
        # Pass through each convolutional layer
        for layer in self.layers_list:
            x = layer(x)
        
        # Flatten the output
        x = self.flatten(x)
        
        # Apply the final fully connected layer
>>>>>>> aa25840 (Laptop commit)
        return self.fc(x)

class StyleGAN(keras.Model):
    def __init__(self, latent_dim=512, style_dim=512, num_layers=8, channels=[512, 256, 128, 64]):
<<<<<<< HEAD
=======
        '''
        Initializes the StyleGAN model, which includes both the Generator and Discriminator networks.

        Args:
            latent_dim (int): Dimensionality of the input latent vector.
            style_dim (int): Dimensionality of the style vector produced by the Mapping Network.
            num_layers (int): Number of layers in the Mapping Network.
            channels (list of int): List of channel sizes for each layer in the Synthesis Network.
        '''
>>>>>>> aa25840 (Laptop commit)
        super(StyleGAN, self).__init__()
        self.generator = Generator(latent_dim, style_dim, num_layers, channels)
        self.discriminator = Discriminator(channels)
        self.latent_dim = latent_dim
<<<<<<< HEAD
    
    def compile(self, g_optimizer, d_optimizer, loss_fn):
=======

    def compile(self, g_optimizer, d_optimizer, loss_fn):
        '''
        Compiles the StyleGAN model with optimizers and loss function.

        Args:
            g_optimizer (tf.keras.optimizers.Optimizer): Optimizer for the generator.
            d_optimizer (tf.keras.optimizers.Optimizer): Optimizer for the discriminator.
            loss_fn (callable): Loss function to use for training.
        '''
>>>>>>> aa25840 (Laptop commit)
        super(StyleGAN, self).compile()
        self.g_optimizer = g_optimizer
        self.d_optimizer = d_optimizer
        self.loss_fn = loss_fn
<<<<<<< HEAD
    
    def train_step(self, real_images):
        batch_size = tf.shape(real_images)[0]
        latent_vectors = tf.random.normal([batch_size, self.latent_dim])
        
        with tf.GradientTape() as g_tape, tf.GradientTape() as d_tape:
            fake_images = self.generator(latent_vectors)
            
            real_output = self.discriminator(real_images)
            fake_output = self.discriminator(fake_images)
            
            g_loss = self.loss_fn(tf.ones_like(fake_output), fake_output)
            d_loss = self.loss_fn(tf.ones_like(real_output), real_output) + \
                     self.loss_fn(tf.zeros_like(fake_output), fake_output)
        
        g_gradients = g_tape.gradient(g_loss, self.generator.trainable_variables)
        d_gradients = d_tape.gradient(d_loss, self.discriminator.trainable_variables)
        
        self.g_optimizer.apply_gradients(zip(g_gradients, self.generator.trainable_variables))
        self.d_optimizer.apply_gradients(zip(d_gradients, self.discriminator.trainable_variables))
        
        return {"g_loss": g_loss, "d_loss": d_loss}

# Usage example
latent_dim = 512
style_dim = 512
num_layers = 8
channels = [512, 256, 128, 64]
batch_size = 32

model = StyleGAN(latent_dim, style_dim, num_layers, channels)
model.compile(
    g_optimizer=keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.0, beta_2=0.99, epsilon=1e-8),
    d_optimizer=keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.0, beta_2=0.99, epsilon=1e-8),
    loss_fn=keras.losses.BinaryCrossentropy(from_logits=True)
)

# Generate dummy data for demonstration
dummy_images = tf.random.normal([batch_size, 64, 64, 3])

# Train for one step (in practice, you would train for many steps)
model.train_step(dummy_images)

print("Training step completed")

# Generate a sample image
sample_latent = tf.random.normal([1, latent_dim])
generated_image = model.generator(sample_latent)
print("Sample image generated")
=======

    @tf.function
    def train_step(self, real_images):
        '''
        Performs a single training step, updating the generator and discriminator.

        Args:
            real_images (tf.Tensor): Batch of real images.

        Returns:
            dict: Dictionary containing the generator loss and discriminator loss.
        '''
        batch_size = tf.shape(real_images)[0]
        latent_vectors = tf.random.normal([batch_size, self.latent_dim])

        with tf.GradientTape() as g_tape, tf.GradientTape() as d_tape:
            fake_images = self.generator(latent_vectors)

            real_output = self.discriminator(real_images)
            fake_output = self.discriminator(fake_images)

            g_loss = self.loss_fn(tf.ones_like(fake_output), fake_output)
            d_loss = self.loss_fn(tf.ones_like(real_output), real_output) + \
                     self.loss_fn(tf.zeros_like(fake_output), fake_output)

        g_gradients = g_tape.gradient(g_loss, self.generator.trainable_variables)
        d_gradients = d_tape.gradient(d_loss, self.discriminator.trainable_variables)

        self.g_optimizer.apply_gradients(zip(g_gradients, self.generator.trainable_variables))
        self.d_optimizer.apply_gradients(zip(d_gradients, self.discriminator.trainable_variables))

        return {"g_loss": g_loss, "d_loss": d_loss}


>>>>>>> aa25840 (Laptop commit)
