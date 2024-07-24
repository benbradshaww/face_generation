import tensorflow as tf
from tensorflow import keras
import numpy as np

class MappingNetwork(keras.Model):
    def __init__(self, latent_dim, style_dim, num_layers):
        super(MappingNetwork, self).__init__()
        self.network = keras.Sequential([
            keras.layers.Dense(style_dim, activation='relu')
            for _ in range(num_layers)
        ])
    
    def call(self, z):
        return self.network(z)

class AdaIN(keras.layers.Layer):
    def __init__(self):
        super(AdaIN, self).__init__()
    
    def call(self, x, style):
        mean = tf.reduce_mean(x, axis=[1, 2], keepdims=True)
        std = tf.math.reduce_std(x, axis=[1, 2], keepdims=True) + 1e-8
        y = (x - mean) / std
        
        style = tf.reshape(style, [-1, 1, 1, style.shape[-1]])
        return y * style[:, :, :, :x.shape[-1]] + style[:, :, :, x.shape[-1]:]

class StyleLayer(keras.layers.Layer):
    def __init__(self, filters, kernel_size, upsample=False):
        super(StyleLayer, self).__init__()
        self.upsample = upsample
        self.conv = keras.layers.Conv2D(filters, kernel_size, padding='same')
        self.adain = AdaIN()
        self.activation = keras.layers.LeakyReLU(0.2)
    
    def call(self, x, style):
        if self.upsample:
            x = tf.image.resize(x, (x.shape[1]*2, x.shape[2]*2), method='bilinear')
        x = self.conv(x)
        x = self.adain(x, style)
        return self.activation(x)

class Generator(keras.Model):
    def __init__(self, latent_dim, style_dim, num_layers, channels):
        super(Generator, self).__init__()
        self.mapping = MappingNetwork(latent_dim, style_dim, num_layers)
        self.initial_const = tf.Variable(tf.random.normal([1, 4, 4, channels[0]]))
        self.style_layers = [
            StyleLayer(ch, 3, upsample=True) for ch in channels[1:]
        ]
        self.to_rgb = keras.layers.Conv2D(3, 1, activation='tanh')
    
    def call(self, z):
        w = self.mapping(z)
        x = tf.tile(self.initial_const, [tf.shape(z)[0], 1, 1, 1])
        
        for layer in self.style_layers:
            x = layer(x, w)
        
        return self.to_rgb(x)

class Discriminator(keras.Model):
    def __init__(self, channels):
        super(Discriminator, self).__init__()
        self.layers_list = [
            keras.layers.Conv2D(ch, 3, strides=2, padding='same', activation='leaky_relu')
            for ch in reversed(channels)
        ]
        self.flatten = keras.layers.Flatten()
        self.fc = keras.layers.Dense(1)
    
    def call(self, x):
        for layer in self.layers_list:
            x = layer(x)
        x = self.flatten(x)
        return self.fc(x)

class StyleGAN(keras.Model):
    def __init__(self, latent_dim=512, style_dim=512, num_layers=8, channels=[512, 256, 128, 64]):
        super(StyleGAN, self).__init__()
        self.generator = Generator(latent_dim, style_dim, num_layers, channels)
        self.discriminator = Discriminator(channels)
        self.latent_dim = latent_dim
    
    def compile(self, g_optimizer, d_optimizer, loss_fn):
        super(StyleGAN, self).compile()
        self.g_optimizer = g_optimizer
        self.d_optimizer = d_optimizer
        self.loss_fn = loss_fn
    
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