import cv2
import numpy as np
from cv_utils import *
import os

import matplotlib.pyplot as plt

# Global variables for Harris parameters
blockSize = 8
ksize = 3
k = 0.05
corner_threshold = 0.1

# Global variables for SIFT parameters
nfeatures = 0
nOctaveLayers = 3
contrastThreshold = 0.2
edgeThreshold = 10
sigma = 1.0

# Function to update Harris parameters
def update_harris_params(x):
    global blockSize, ksize, k, corner_threshold
    blockSize = cv2.getTrackbarPos('Block Size', 'Parameters')
    if blockSize < 2:
        blockSize = 2
    ksize = cv2.getTrackbarPos('K Size', 'Parameters')
    k = cv2.getTrackbarPos('K', 'Parameters') / 100.0
    corner_threshold = cv2.getTrackbarPos('Corner Threshold', 'Parameters') / 100.0
    if corner_threshold == 0:
        corner_threshold = 0.01

# Function to update SIFT parameters
def update_sift_params(x):
    global nfeatures, nOctaveLayers, contrastThreshold, edgeThreshold, sigma
    nfeatures = cv2.getTrackbarPos('N Features', 'Parameters')
    nOctaveLayers = cv2.getTrackbarPos('N Octave Layers', 'Parameters')
    contrastThreshold = cv2.getTrackbarPos('Contrast Threshold', 'Parameters') / 100.0
    edgeThreshold = cv2.getTrackbarPos('Edge Threshold', 'Parameters')
    sigma = cv2.getTrackbarPos('Sigma', 'Parameters')
    if sigma == 0:
        sigma = 1 
    print(sigma)


# Create a window for displaying the parameters
cv2.namedWindow('Parameters')

# Create trackbars for Harris parameters
cv2.createTrackbar('Block Size', 'Parameters', blockSize, 20, update_harris_params)
cv2.createTrackbar('K Size', 'Parameters', ksize, 20, update_harris_params)
cv2.createTrackbar('K', 'Parameters', int(k * 100), 100, update_harris_params)
cv2.createTrackbar('Corner Threshold', 'Parameters', int(corner_threshold * 100), 100, update_harris_params)

# Create trackbars for SIFT parameters
cv2.createTrackbar('N Features', 'Parameters', nfeatures, 1000, update_sift_params)
cv2.createTrackbar('N Octave Layers', 'Parameters', nOctaveLayers, 10, update_sift_params)
cv2.createTrackbar('Contrast Threshold', 'Parameters', int(contrastThreshold * 100), 100, update_sift_params)
cv2.createTrackbar('Edge Threshold', 'Parameters', edgeThreshold, 100, update_sift_params)
cv2.createTrackbar('Sigma', 'Parameters', int(sigma*10), 20, update_sift_params)

# Load the input frame
frame = cv2.imread('UnityHall.jpg')

display_harris = True
display_refined_harris = False

while True:
    # Copy the frame for display
    display_frame = frame.copy()
    ksize = max(ksize, 3)  # Ensure ksize is at least 3
    ksize = ksize + 1 if ksize % 2 == 0 else ksize  # Make ksize odd if it's even
    if display_harris:
        # Perform Harris corner detection
        harris_frame, refined_frame = harris(frame, blockSize, ksize, k, corner_threshold)
        if display_refined_harris:
            cv2.destroyWindow('SIFT Feature Detection')  # Close SIFT window if open
            cv2.destroyWindow('Refined Harris Corner Detection')  # Close refined Harris window if open
            cv2.imshow('Harris Corner Detection', refined_frame)
           
        else:
            cv2.destroyWindow('SIFT Feature Detection')  # Close SIFT window if open
            cv2.destroyWindow('Harris Corner Detection')  # Close SIFT window if open
            cv2.imshow('Refined Harris Corner Detection', harris_frame)
            
    else:
        # Perform SIFT feature detection
        sift_frame = sift(frame, nfeatures, nOctaveLayers, contrastThreshold, edgeThreshold, sigma)
        cv2.destroyWindow('Harris Corner Detection')  # Close Harris window if open
        cv2.destroyWindow('Refined Harris Corner Detection')
        cv2.imshow('SIFT Feature Detection', sift_frame)
        
    

    # Check for keypress
    key = cv2.waitKey(1) & 0xFF
    if key == ord('h'):
        display_harris = True
        display_refined_harris = False
    elif key == ord('s'):
        display_harris = False
        display_refined_harris = False
    elif key == ord('r'):
        display_harris = True
        display_refined_harris = True
    elif key == 27:  # Esc key
        break

cv2.destroyAllWindows()
