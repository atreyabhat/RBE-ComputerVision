import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, losses
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Model

# Load and preprocess the Fashion MNIST dataset
(x_train, _), (x_test, _) = fashion_mnist.load_data()
x_train = x_train.astype('float32') / 255.0  # Normalize training images
x_test = x_test.astype('float32') / 255.0    # Normalize test images

print(f"Training data shape: {x_train.shape}")
print(f"Test data shape: {x_test.shape}")

# Add noise to the images
noise_factor = 0.2
x_train_noisy = x_train + noise_factor * tf.random.normal(shape=x_train.shape)
x_test_noisy = x_test + noise_factor * tf.random.normal(shape=x_test.shape)

# Clip pixel values to be in the [0, 1] range
x_train_noisy = tf.clip_by_value(x_train_noisy, clip_value_min=0.0, clip_value_max=1.0)
x_test_noisy = tf.clip_by_value(x_test_noisy, clip_value_min=0.0, clip_value_max=1.0)

# Define the denoising autoencoder model
class Denoise(Model):
    def __init__(self):
        super(Denoise, self).__init__()
        # Encoder: Extracts features from noisy images
        self.encoder = tf.keras.Sequential([
            layers.Input(shape=(28, 28, 1)),
            layers.Conv2D(16, (3, 3), activation='relu', padding='same', strides=2),
            layers.Conv2D(8, (3, 3), activation='relu', padding='same', strides=2)
        ])

        # Decoder: Reconstructs clean images from encoded features
        self.decoder = tf.keras.Sequential([
            layers.Conv2DTranspose(8, kernel_size=3, strides=2, activation='relu', padding='same'),
            layers.Conv2DTranspose(16, kernel_size=3, strides=2, activation='relu', padding='same'),
            layers.Conv2D(1, kernel_size=3, activation='sigmoid', padding='same')
        ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# Instantiate and compile the denoising autoencoder
autoencoder = Denoise()
autoencoder.compile(optimizer='adam', loss=losses.MeanSquaredError())

# Train the autoencoder
autoencoder.fit(x_train_noisy, x_train,
                epochs=10,
                shuffle=True,
                validation_data=(x_test_noisy, x_test))

# Encode and decode noisy images
encoded_imgs = autoencoder.encoder(x_test_noisy).numpy()
decoded_imgs = autoencoder.decoder(encoded_imgs).numpy()

# Plot noisy and denoised images
n = 10  # Number of images to display
plt.figure(figsize=(20, 4))

for i in range(n):
    # Display noisy images
    ax = plt.subplot(2, n, i + 1)
    plt.title("Noisy Image")
    plt.imshow(tf.squeeze(x_test_noisy[i]), cmap='gray')
    ax.axis('off')

    # Display denoised images
    bx = plt.subplot(2, n, i + 1 + n)
    plt.title("Denoised Image")
    plt.imshow(tf.squeeze(decoded_imgs[i]), cmap='gray')
    bx.axis('off')

plt.show()
