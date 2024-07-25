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

# Define the autoencoder model class
class BasicAutoencoder(Model):
    def __init__(self, latent_dim, input_shape):
        super(BasicAutoencoder, self).__init__()
        self.latent_dim = latent_dim
        self.input_shape = input_shape

        # Encoder: Flattens the input and compresses it into a latent space
        self.encoder = tf.keras.Sequential([
            layers.Flatten(),
            layers.Dense(latent_dim, activation='relu')
        ])

        # Decoder: Expands the latent space representation back to the original shape
        self.decoder = tf.keras.Sequential([
            layers.Dense(np.prod(input_shape), activation='sigmoid'),
            layers.Reshape(input_shape)
        ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# Set model parameters
input_shape = x_test.shape[1:]  # Image shape without batch dimension
latent_dim = 128  # Dimension of the latent space

# Instantiate and compile the autoencoder model
autoencoder = BasicAutoencoder(latent_dim, input_shape)
autoencoder.compile(optimizer='adam', loss=losses.MeanSquaredError())

# Train the autoencoder
autoencoder.fit(x_train, x_train,
                epochs=20,
                shuffle=True,
                validation_data=(x_test, x_test))

# Encode and decode images
encoded_imgs = autoencoder.encoder(x_test).numpy()
decoded_imgs = autoencoder.decoder(encoded_imgs).numpy()

# Plot original and reconstructed images
n = 10  # Number of images to display
plt.figure(figsize=(20, 4))

for i in range(n):
    # Display original images
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i], cmap='gray')
    plt.title("Original")
    ax.axis('off')

    # Display reconstructed images
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i], cmap='gray')
    plt.title("Reconstructed")
    ax.axis('off')

plt.show()
