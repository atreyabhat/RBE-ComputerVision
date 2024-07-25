import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

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
test_labels = np.repeat(k, 250)[:, np.newaxis]  # Ensure test_labels are different from train_labels

accuracies = []

knn = cv2.ml.KNearest_create()
knn.train(train, cv2.ml.ROW_SAMPLE, train_labels)

# Test for k = 1-9
k_max = 10
for k in range(1, k_max):
    _, result, _, _ = knn.findNearest(test, k=k)

    accuracy = accuracy_score(test_labels, result)
    accuracies.append(accuracy * 100)
    # print(f"Accuracy for k = {k}: {accuracy * 100:.2f}%")

# Plotting
k_values = range(1, k_max)
plt.plot(k_values, accuracies, marker='o')
plt.xlabel('k')
plt.ylabel('Accuracy (%)')
plt.title('Accuracy vs. k')
plt.xticks(k_values)
plt.grid(True)
plt.show()

