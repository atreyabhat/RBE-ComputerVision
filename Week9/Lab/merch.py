import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.applications import VGG19, InceptionV3
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import cv2, numpy as np


def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img


# Load the dataset
data_dir = 'MerchData/MerchData'

train_ds = image_dataset_from_directory(
    data_dir,
    label_mode='int',
    image_size=(224, 224),
    batch_size=4,
    validation_split=0.3,
    subset="training",
    seed=123
)

val_ds = image_dataset_from_directory(
    data_dir,
    label_mode='int',
    image_size=(224, 224),
    batch_size=4,
    validation_split=0.3,
    subset="validation",
    seed=123
)


num_classes = len(train_ds.class_names)

# Display class names
print("Class names:", train_ds.class_names)

# Data augmentation and preprocessing
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
    tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
    tf.keras.layers.experimental.preprocessing.Rescaling(1./255)
])

train_ds = train_ds.map(lambda x, y: (data_augmentation(x, training=True), y))
val_ds = val_ds.map(lambda x, y: (data_augmentation(x, training=False), y))

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# Load pretrained network
base_model = VGG19(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
# base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze initial layers
for layer in base_model.layers:
    layer.trainable = False


# Add new layers on top of the base model
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model
model.compile(optimizer=Adam(learning_rate=1e-4),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])



##############################################################

initial_epochs = 15
fine_tune_epochs = 10
layers_to_freeze = base_model.layers[:10]

##############################################################




history_base = model.fit(train_ds,
                         epochs=initial_epochs,
                         validation_data=val_ds)


for layer in layers_to_freeze:
    layer.trainable = True

model.compile(optimizer=Adam(learning_rate=1e-5),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history_fine = model.fit(train_ds,
                         epochs=initial_epochs + fine_tune_epochs,
                         initial_epoch=history_base.epoch[-1],
                         validation_data=val_ds)

# Plot training history
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(history_base.history['accuracy'] + history_fine.history['accuracy'])
plt.plot(history_base.history['val_accuracy'] + history_fine.history['val_accuracy'])
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Training Accuracy', 'Validation Accuracy'])

plt.subplot(1, 2, 2)
plt.plot(history_base.history['loss'] + history_fine.history['loss'])
plt.plot(history_base.history['val_loss'] + history_fine.history['val_loss'])
plt.title('Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Training Loss', 'Validation Loss'])

plt.show()


# random_images = ['cap.jpeg', 'torch.jpeg', 'cube.jpeg', 'pen.png']

# class_names =  ['Cap', 'Cube', 'Playing Cards', 'Screwdriver', 'Torch']
# print(class_names)

# plt.figure(figsize=(10, 10))

# for i, img_path in enumerate(random_images):
#     img = preprocess_image(img_path)
#     predictions = model.predict(img)
#     predicted_class = class_names[np.argmax(predictions[0])]
    
#     # Read the original image for plotting
#     img_to_plot = cv2.imread(img_path)
#     img_to_plot = cv2.cvtColor(img_to_plot, cv2.COLOR_BGR2RGB)  
    
#     plt.subplot(1, len(random_images), i + 1)
#     plt.imshow(img_to_plot)
#     plt.title(f"Predicted: {predicted_class}")
#     plt.axis('off')  # Hide axes

# plt.tight_layout()
# plt.show()

# model.save('merch_model.h5')