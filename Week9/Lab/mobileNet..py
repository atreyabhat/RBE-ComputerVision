import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.layers import Dropout
import matplotlib.pyplot as plt
import pathlib

# Download and prepare the dataset
data_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
data_root_orig = tf.keras.utils.get_file(origin=data_url, fname='flower_photos', untar=True)
data_root = pathlib.Path(data_root_orig)

# Image preprocessing parameters
image_size = (160, 160)
batch_size = 32

# Create a dataset
train_ds = tf.keras.preprocessing.image_dataset_from_directory(data_root,
                                                               validation_split=0.2,
                                                               subset="training",
                                                               seed=123,
                                                               image_size=image_size,
                                                               batch_size=batch_size)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(data_root,
                                                             validation_split=0.2,
                                                             subset="validation",
                                                             seed=123,
                                                             image_size=image_size,
                                                             batch_size=batch_size)

# Data augmentation
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip('horizontal'),
    tf.keras.layers.RandomRotation(0.2),
    tf.keras.layers.RandomZoom(0.2),
])

# Preprocess the datasets
def preprocess(image, label):
    image = tf.image.resize(image, image_size)
    image = image / 255.0
    return image, label

train_ds = train_ds.map(preprocess)
val_ds = val_ds.map(preprocess)

# Add data augmentation to the training dataset
train_ds = train_ds.map(lambda x, y: (data_augmentation(x, training=True), y))

# Cache and prefetch datasets for performance
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# Step 1: Train a base model with data augmentation
base_model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=image_size + (3,)),
    tf.keras.layers.Conv2D(32, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(64, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(128, 3, activation='relu'),
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(5, activation='softmax')
])

# Compile the base model
base_model.compile(optimizer='adam',
                   loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                   metrics=['accuracy'])

# Train the base model
initial_epochs = 25
history_base = base_model.fit(train_ds,
                              epochs=initial_epochs,
                              validation_data=val_ds)

# Step 2: Fine-tune the MobileNetV2 model with dropout
base_model = tf.keras.applications.MobileNetV2(input_shape=image_size + (3,),
                                               include_top=False,
                                               weights='imagenet')

# Fine-tuning the model
base_model.trainable = True
fine_tune_at = 120

# Freeze all the layers before the `fine_tune_at` layer
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

# Add dropout layer to reduce overfitting
x = base_model.output
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(1024, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = tf.keras.layers.Dense(5, activation='softmax')(x)
model = tf.keras.Model(inputs=base_model.input, outputs=predictions)

# Compile the model for fine-tuning
base_learning_rate = 0.0001
model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              optimizer=tf.keras.optimizers.RMSprop(learning_rate=base_learning_rate/10),
              metrics=['accuracy'])

model.summary()

# Fine-tuning epochs
fine_tune_epochs = 25
total_epochs = initial_epochs + fine_tune_epochs

history_fine = model.fit(train_ds,
                         epochs=total_epochs,
                         initial_epoch=history_base.epoch[-1],
                         validation_data=val_ds)

# Plot fine-tuning results
acc = history_base.history['accuracy']
val_acc = history_base.history['val_accuracy']
acc += history_fine.history['accuracy']
val_acc += history_fine.history['val_accuracy']

loss = history_base.history['loss']
val_loss = history_base.history['val_loss']
loss += history_fine.history['loss']
val_loss += history_fine.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.plot([initial_epochs-1, initial_epochs-1],
         plt.ylim(), label='Start Fine Tuning')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.plot([initial_epochs-1, initial_epochs-1],
         plt.ylim(), label='Start Fine Tuning')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()

# Save the fine-tuned model
model.save('mobileNet_fine_tuned_model.h5')


#achieves 90% val accuracy