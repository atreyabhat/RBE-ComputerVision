import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import datetime

# Load the MNIST dataset
mnist = keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Normalize the pixel values to be between 0 and 1
train_images = train_images / 255.0
test_images = test_images / 255.0

# Add channel dimension
train_images = train_images[..., tf.newaxis]
test_images = test_images[..., tf.newaxis]

# Sequential API (Deep network)
def create_sequential_model():
    model = keras.Sequential([
        keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        keras.layers.MaxPooling2D(2, 2),
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(10, activation='softmax')
    ])
    return model

# Functional API (Deep network)
def create_deep_functional_model():
    inputs = keras.Input(shape=(28, 28, 1))
    x = keras.layers.Conv2D(64, (3, 3), activation='relu')(inputs)
    x = keras.layers.MaxPooling2D(2, 2)(x)
    x = keras.layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = keras.layers.MaxPooling2D(2, 2)(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(128, activation='relu')(x)
    outputs = keras.layers.Dense(10, activation='softmax')(x)
    model = keras.Model(inputs, outputs)
    return model

# Function to plot accuracies
def plot_accuracy(histories, title):
    plt.figure(figsize=(12, 6))
    for opt, history in histories.items():
        plt.plot(history.history['accuracy'], label=f'Train Accuracy ({opt})')
        plt.plot(history.history['val_accuracy'], label=f'Validation Accuracy ({opt})')
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

# Training configurations
optimizers = ['adam', 'sgd', 'rmsprop']
epochs = 10

# Sequential model training
histories = {}
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

for opt in optimizers:
    print(f"\nTraining Sequential model with {opt} optimizer")
    seq_model = create_sequential_model()
    seq_model.compile(optimizer=opt, 
                      loss='sparse_categorical_crossentropy', 
                      metrics=['accuracy'])
    histories[opt] = seq_model.fit(train_images, train_labels, epochs=epochs, validation_split=0.1, callbacks=[tensorboard_callback])
    seq_test_loss, seq_test_acc = seq_model.evaluate(test_images, test_labels)
    print(f'Sequential API Test accuracy with {opt}: {seq_test_acc}')

# Plot accuracies for Sequential model
plot_accuracy(histories, 'Sequential API with Different Optimizers')

# Deep Functional model training
histories = {}
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

for opt in optimizers:
    print(f"\nTraining Deep Functional model with {opt} optimizer")
    func_model_deep = create_deep_functional_model()
    func_model_deep.compile(optimizer=opt, 
                            loss='sparse_categorical_crossentropy', 
                            metrics=['accuracy'])
    histories[opt] = func_model_deep.fit(train_images, train_labels, epochs=epochs, validation_split=0.1, callbacks=[tensorboard_callback])
    func_deep_test_loss, func_deep_test_acc = func_model_deep.evaluate(test_images, test_labels)
    print(f'Functional API Deep network Test accuracy with {opt}: {func_deep_test_acc}')

# Plot accuracies for Deep Functional model
plot_accuracy(histories, 'Functional API Deep Network with Different Optimizers')

