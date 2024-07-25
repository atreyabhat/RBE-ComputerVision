#########################################
## K 1-9 and various splits

import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

img = cv2.imread('data/digits.png', cv2.IMREAD_GRAYSCALE)

# Each digit is a 20x20 image
cells = [np.hsplit(row, 100) for row in np.vsplit(img, 50)]

# Convert the list of lists into a numpy array
x = np.array(cells)

# Prepare the data by reshaping each 20x20 image into a single row of 400 pixels
data = x.reshape(-1, 400).astype(np.float32)
labels = np.repeat(np.arange(10), 500)

# Define train/test splits from 10% to 90% for training data
train_sizes = np.arange(0.1, 1.0, 0.1)
k_values = np.arange(1, 10)
accuracy_results = np.zeros((len(train_sizes), len(k_values)))

for i, train_size in enumerate(train_sizes):
    X_train, X_test, y_train, y_test = train_test_split(data, labels, train_size=train_size, random_state=42)

    for j, k in enumerate(k_values):
        knn = cv2.ml.KNearest_create()
        knn.train(X_train, cv2.ml.ROW_SAMPLE, y_train)
        _, result, _, _ = knn.findNearest(X_test, k=k)
        
        # Calculate accuracy
        accuracy = accuracy_score(y_test, result)
        accuracy_results[i, j] = accuracy

# Plotting results
plt.figure(figsize=(10, 6))
for j, k in enumerate(k_values):
    plt.plot(train_sizes, accuracy_results[:, j], marker='o', label=f'k={k}')

plt.xlabel('Training Size')
plt.ylabel('Accuracy')
plt.title('Accuracy vs. Training Size for Different k Values')
plt.legend(title='Number of Neighbors (k)')
plt.grid(True)
plt.tight_layout()
plt.show()

