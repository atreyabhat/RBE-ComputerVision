import tensorflow as tf
from tensorflow import keras
import cv2
import numpy as np

# Load the pre-trained model from H5 file
model = keras.models.load_model('mobileNet_fine_tuned_model.h5')

# Load class names (assuming you have a list of class names)
class_names = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']

def preprocess_frame(frame):
    img = cv2.resize(frame, (160, 160))  # Resize frame to the input size expected by the model
    img = img / 255.0  # Normalize pixel values
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# Capture webcam input
cap = cv2.VideoCapture(2)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the frame
    preprocessed_frame = preprocess_frame(frame)

    # Make predictions
    predictions = model.predict(preprocessed_frame)
    score = tf.nn.softmax(predictions[0])

    # Draw the label and confidence on the frame
    label = f'{class_names[np.argmax(score)]}: {100 * np.max(score):.2f}%'
    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('Webcam Input', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
