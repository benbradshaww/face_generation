import tensorflow as tf
import tensorflow_addons as tfa

class DDPM(tf.keras.Model):
    def __init__(self, time_steps=1000, img_size=64, num_channels=3):
        super(DDPM, self).__init__()
        self.time_steps = time_steps
        self.img_size = img_size
        self.num_channels = num_channels
        
        # Define beta schedule
        self.betas = tf.linspace(1e-4, 0.02, time_steps)
        self.alphas = 1 - self.betas
        self.alphas_cumprod = tf.math.cumprod(self.alphas)
        
        # U-Net architecture
        self.down1 = self.conv_block(64)
        self.down2 = self.conv_block(128)
        self.down3 = self.conv_block(256)
        self.down4 = self.conv_block(512)
        
        self.up1 = self.conv_block(256)
        self.up2 = self.conv_block(128)
        self.up3 = self.conv_block(64)
        self.final_conv = tf.keras.layers.Conv2D(num_channels, 3, padding='same')
        
        # Time embedding
        self.time_mlp = tf.keras.Sequential([
            tf.keras.layers.Dense(256, activation='swish'),
            tf.keras.layers.Dense(512, activation='swish'),
        ])

        # Additional layers for time embedding
        self.time_emb1 = tf.keras.layers.Dense(64)
        self.time_emb2 = tf.keras.layers.Dense(128)
        self.time_emb3 = tf.keras.layers.Dense(256)
        self.time_emb4 = tf.keras.layers.Dense(512)

    def compile(self, optimizer=tf.keras.optimizers.Adam(learning_rate=2e-4), loss=tf.keras.losses.MeanSquaredError()):
        super(DDPM, self).compile()
        self.optimizer = optimizer
        self.loss_fn = loss

    def conv_block(self, filters):
        return tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters, 3, padding='same'),
            tfa.layers.GroupNormalization(groups=32),
            tf.keras.layers.Activation('swish'),
            tf.keras.layers.Conv2D(filters, 3, padding='same'),
            tfa.layers.GroupNormalization(groups=32),
            tf.keras.layers.Activation('swish')
        ])

    def call(self, x, t):
        # Ensure t is the correct shape [batch_size, 1]
        t = tf.reshape(t, [-1, 1])
        t = self.time_mlp(t)
        
        # Down-sampling
        x1 = self.down1(x)
        x1 += self.time_emb1(t)[:, tf.newaxis, tf.newaxis, :]
        x2 = self.down2(tf.keras.layers.MaxPooling2D()(x1))
        x2 += self.time_emb2(t)[:, tf.newaxis, tf.newaxis, :]
        x3 = self.down3(tf.keras.layers.MaxPooling2D()(x2))
        x3 += self.time_emb3(t)[:, tf.newaxis, tf.newaxis, :]
        x4 = self.down4(tf.keras.layers.MaxPooling2D()(x3))
        x4 += self.time_emb4(t)[:, tf.newaxis, tf.newaxis, :]
        
        # Up-sampling
        x = tf.keras.layers.UpSampling2D()(x4)
        x = self.up1(tf.concat([x, x3], axis=-1))
        x += self.time_emb3(t)[:, tf.newaxis, tf.newaxis, :]
        x = tf.keras.layers.UpSampling2D()(x)
        x = self.up2(tf.concat([x, x2], axis=-1))
        x += self.time_emb2(t)[:, tf.newaxis, tf.newaxis, :]
        x = tf.keras.layers.UpSampling2D()(x)
        x = self.up3(tf.concat([x, x1], axis=-1))
        x += self.time_emb1(t)[:, tf.newaxis, tf.newaxis, :]
        
        return self.final_conv(x)

    def diffusion_schedule(self, x_0):
        batch_size = tf.shape(x_0)[0]
        t = tf.random.uniform([batch_size], minval=0, maxval=self.time_steps, dtype=tf.int32)
        alpha_cumprod = tf.gather(self.alphas_cumprod, t)
        alpha_cumprod = tf.reshape(alpha_cumprod, [batch_size, 1, 1, 1])
        
        noise = tf.random.normal(tf.shape(x_0))
        x_t = tf.sqrt(alpha_cumprod) * x_0 + tf.sqrt(1 - alpha_cumprod) * noise
        
        return x_t, noise, t

    def train_step(self, x):
        with tf.GradientTape() as tape:
            x_t, noise, t = self.diffusion_schedule(x)
            predicted_noise = self(x_t, t)
            loss = self.loss_fn(noise, predicted_noise)
        
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        
        return {"loss": loss}

    def generate_images(self, num_images=1):
        x = tf.random.normal([num_images, self.img_size, self.img_size, self.num_channels])
        
        for i in reversed(range(self.time_steps)):
            t = tf.ones([num_images, i])
            predicted_noise = self(x, t)
            alpha = self.alphas[i]
            alpha_cumprod = self.alphas_cumprod[i]
            beta = self.betas[i]

            if i > 0:
                noise = tf.random.normal(tf.shape(x))
            else:
                noise = tf.zeros_like(x)
            
            x = 1 / tf.sqrt(alpha) * (x - ((1 - alpha) / (tf.sqrt(1 - alpha_cumprod))) * predicted_noise) + tf.sqrt(beta) * noise
        
        return (x + 1) / 2  # Scale to [0, 1]

