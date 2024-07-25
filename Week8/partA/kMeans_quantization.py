import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image = cv2.imread('data/nature.png')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB

# Prepare the data
Z = image.reshape((-1, 3))  # Reshape to a 2D array of pixels

# Convert to float32
Z = np.float32(Z)

# Criteria for K-Means
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)

# Number of clusters to try
k_values = [2, 3, 5, 10, 20, 40]
quantized_images = []

for k in k_values:
    # Apply K-Means clustering
    _, labels, centers = cv2.kmeans(Z, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    # Convert centers to uint8
    centers = np.uint8(centers)
    
    # Map the labels to the centers
    quantized_img = centers[labels.flatten()]
    
    # Reshape back to the original image shape
    quantized_img = quantized_img.reshape(image.shape)
    
    quantized_images.append(quantized_img)

# Plot the results
plt.figure(figsize=(12, 8))

# Plot original image
plt.subplot(2, 4, 1)
plt.imshow(image)
plt.title('Original')
plt.axis('off')

# Plot quantized images
for i, k in enumerate(k_values):
    plt.subplot(2, 4, i+2)
    plt.imshow(quantized_images[i])
    plt.title(f'k={k}')
    plt.axis('off')

plt.tight_layout()
plt.show()
