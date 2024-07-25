import cv2
import numpy as np

# Load the digits image
img = cv2.imread('data/digits.png', cv2.IMREAD_GRAYSCALE)

# Each digit is a 20x20 image
cells = [np.hsplit(row, 100) for row in np.vsplit(img, 50)]

# Convert the list of lists into a numpy array
x = np.array(cells)

# Prepare the data by reshaping each 20x20 image into a single row of 400 pixels
train = x[:, :50].reshape(-1, 400).astype(np.float32)  # Training data
test = x[:, 50:100].reshape(-1, 400).astype(np.float32)  # Testing data

# Create labels for the data
k = np.arange(10)
train_labels = np.repeat(k, 250)[:, np.newaxis]
test_labels = train_labels


knn = cv2.ml.KNearest_create()

knn.train(train, cv2.ml.ROW_SAMPLE, train_labels)

# Test 
ret, result, neighbours, dist = knn.findNearest(test, k=5)

# Calculate the accuracy
matches = result == test_labels
correct = np.count_nonzero(matches)
accuracy = correct * 100.0 / result.size

print(f"Accuracy: {accuracy}%")
np.savez('knn_vanilla.npz',train=train, train_labels=train_labels)
