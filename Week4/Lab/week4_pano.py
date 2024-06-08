import cv2
import numpy as np
import os
from cv_utils import *

import matplotlib.pyplot as plt

# Define the folder path where the images are stored
folder_path = 'data/boston/'

# Get the list of image files in the folder
image_files = os.listdir(folder_path)

# Create an empty list to store the loaded images
images = []

# Load each image and append it to the list
for file_name in image_files:
    image_path = os.path.join(folder_path, file_name)
    image = cv2.imread(image_path)
    images.append(image)

# Initialize the panorama with the first image
panorama = images[0]
print(panorama.shape)

# Iterate through the remaining images
for i in range(1, len(images)):
    img1 = images[i]
    img2 = panorama

    # Perform SIFT feature matching between the two images
    res_img, _, M = sift_feature_matching(img2, img1)

    # Get the dimensions of the images
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    # Define the corner points of the images
    corner_pts1 = np.float32([[0,0],[0,h1],[w1,h1],[w1,0]]).reshape(-1,1,2)
    corner_pts2 = np.float32([[0,0],[0,h2],[w2,h2],[w2,0]]).reshape(-1,1,2)

    # Warp the second image's corner points using the transformation matrix
    corner_pts2_warped = cv2.perspectiveTransform(corner_pts2, M)

    # Concatenate the corner points of both images
    corners = np.concatenate((corner_pts1, corner_pts2_warped), axis=0)

    # Get the minimum and maximum coordinates of the corners
    [xmin, ymin] = np.int32(corners.min(axis=0).ravel())
    [xmax, ymax] = np.int32(corners.max(axis=0).ravel())

    # Shift the transformation matrix to align the images
    M_shift = np.array([[1,0,-xmin],[0,1,-ymin],[0,0,1]], dtype=np.float32)

    # Calculate the size of the canvas for the panorama
    canvas_size = (xmax-xmin, ymax-ymin)

    # Warp the first and second images to the panorama canvas
    img1_warped = cv2.warpPerspective(img1, M_shift, canvas_size)
    img2_warped = cv2.warpPerspective(img2, M_shift.dot(M), canvas_size)

    # Blend the warped images to create the panorama
    panorama = cv2.addWeighted(img1_warped, 0.5, img2_warped, 0.5, 0)

    # Save the panorama image
    cv2.imwrite('boston_pano.jpg', panorama)

# Display the final panorama
plt.figure(figsize=(20, 10))
plt.imshow(cv2.cvtColor(panorama, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.title('Panorama')
plt.show()

# Display the SIFT feature matching result
plt.figure(figsize=(20, 10))
plt.imshow(res_img)
