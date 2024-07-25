import cv2 as cv
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained CNN model
model = load_model('mnist_cnn_model.h5')

# Function to preprocess the image (resize and normalize)
def preprocess_image(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    _, gray = cv.threshold(gray, 128, 255, cv.THRESH_BINARY_INV)
    resized_image = cv.resize(gray, (28, 28))  # Resize to 28x28
    resized_image = cv.GaussianBlur(resized_image, (5, 5), 0)
    resized_image = resized_image.astype('float32') / 255  # Normalize to [0, 1]
    resized_image = np.expand_dims(resized_image, axis=-1)  # Add channel dimension
    resized_image = np.expand_dims(resized_image, axis=0)  # Add batch dimension
    return resized_image

# Initialize video capture
cap = cv.VideoCapture(2)  # 0 for the default camera

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Preprocess the frame
    processed_image = preprocess_image(frame)
    
    # Predict the digit
    predictions = model.predict(processed_image)
    digit = np.argmax(predictions[0])
    
    # Draw the digit label on the frame
    cv.putText(frame, str(digit), (50, 120), cv.FONT_HERSHEY_SIMPLEX, 5, (0, 255, 0), 3)
    
    # Display the frame
    cv.imshow('Digit Recognition', frame)
    
    # Break the loop on 'q' key press
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv.destroyAllWindows()
