import cv2
import numpy as np
import matplotlib.pyplot as plt
from cv_utils import *
import os


# pts1 = []
# pts2 = []

# Sample points for perspective transformation
pts1 = [[100, 100], [200, 100], [200, 200], [100, 200]]
pts2 = [[100, 100], [400, 100], [400, 300], [100, 300]]

def mouse_callback(event, x, y, flags, param):
    """
    Mouse callback function for selecting points on the image for perspective transformation.

    Parameters:
    - event: The event type (e.g. left button down, left button up, etc.)
    - x: The x-coordinate of the mouse click
    - y: The y-coordinate of the mouse click
    - flags: Additional flags
    - param: Additional parameters

    Returns:
    None
    """
    global pts1, pts2
    frame_copy = frame.copy()
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(pts1) < 4:
            pts1.append([x, y])
            cv2.circle(frame_copy, (x, y), 5, (0, 0, 255), -1)
            cv2.imshow('Original Frame', frame_copy)
        elif len(pts2) < 4:
            pts2.append([x, y])
            cv2.circle(frame_copy, (x, y), 5, (0, 255, 0), -1)
            cv2.imshow('Original Frame', frame_copy)

#########################################################################################################


frame = cv2.imread('UnityHall.jpg')
# frame = cv2.imread('arrows.png')
rotated_frame = rotate_image(frame, 10)
scaled_up_frame = scale_frame(frame, 1.2)
scaled_down_frame = scale_frame(frame, 0.8)
affine_tf_frame = affine_transform(frame, 30)

# cv2.imshow('Original Frame', frame)
# # cv2.setMouseCallback('Original Frame', mouse_callback)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

perspective_tf_frame = perspective_transform(frame, np.float32(pts1), np.float32(pts2))

# cv2.imshow('Perspective Transformed Frame', perspective_tf_frame)
# cv2.imshow('Rotated Frame', rotated_frame)
# cv2.imshow('Scaled Up Frame', scaled_up_frame)
# cv2.imshow('Scaled Down Frame', scaled_down_frame)
# cv2.imshow('Affine Transformed Frame', affine_tf_frame)

cv2.waitKey(0)
cv2.destroyAllWindows()


blockSize = 4
ksize = 3
k = 0.04
corner_threshold = 0.2

harris_frame, refined_harris = harris(frame, blockSize, ksize, k,corner_threshold)
harris_rotated_frame, _ = harris(rotated_frame, blockSize, ksize, k)
harris_scaled_up_frame,_ = harris(scaled_up_frame, blockSize, ksize, k)
harris_scaled_down_frame,_ = harris(scaled_down_frame, blockSize, ksize, k)
harris_affine_tf_frame,_ = harris(affine_tf_frame, blockSize, ksize, k)
harris_perspective_tf_frame,_ = harris(perspective_tf_frame, blockSize, ksize, k)


sift_frame = sift(frame)
sift_rotated_frame = sift(rotated_frame)
sift_scaled_up_frame = sift(scaled_up_frame)
sift_scaled_down_frame = sift(scaled_down_frame)
sift_affine_tf_frame = sift(affine_tf_frame)
sift_perspective_tf_frame = sift(perspective_tf_frame)


# Create a directory to store the results
results_dir = 'results'
os.makedirs(results_dir, exist_ok=True)
# Save images with proper file names
cv2.imwrite(os.path.join(results_dir, 'harris_frame.jpg'), harris_frame)
cv2.imwrite(os.path.join(results_dir, 'refined_harris.jpg'), refined_harris)
cv2.imwrite(os.path.join(results_dir, 'harris_rotated.jpg'), harris_rotated_frame)
cv2.imwrite(os.path.join(results_dir, 'harris_scaled_up.jpg'), harris_scaled_up_frame)
cv2.imwrite(os.path.join(results_dir, 'harris_scaled_down.jpg'), harris_scaled_down_frame)
cv2.imwrite(os.path.join(results_dir, 'harris_affine_tf.jpg'), harris_affine_tf_frame)
cv2.imwrite(os.path.join(results_dir, 'harris_perspective_tf.jpg'), harris_perspective_tf_frame)
cv2.imwrite(os.path.join(results_dir, 'sift_frame.jpg'), sift_frame)
cv2.imwrite(os.path.join(results_dir, 'sift_rotated.jpg'), sift_rotated_frame)
cv2.imwrite(os.path.join(results_dir, 'sift_scaled_up.jpg'), sift_scaled_up_frame)
cv2.imwrite(os.path.join(results_dir, 'sift_scaled_down.jpg'), sift_scaled_down_frame)
cv2.imwrite(os.path.join(results_dir, 'sift_affine_tf.jpg'), sift_affine_tf_frame)
cv2.imwrite(os.path.join(results_dir, 'sift_perspective_tf.jpg'), sift_perspective_tf_frame)
cv2.imwrite(os.path.join(results_dir, 'original_frame.jpg'), cv2.imread('UnityHall.jpg'))
cv2.imwrite(os.path.join(results_dir, 'rotated.jpg'), rotated_frame)
cv2.imwrite(os.path.join(results_dir, 'scaled_up.jpg'), scaled_up_frame)
cv2.imwrite(os.path.join(results_dir, 'scaled_down.jpg'), scaled_down_frame)
cv2.imwrite(os.path.join(results_dir, 'affine_tf.jpg'), affine_tf_frame)
cv2.imwrite(os.path.join(results_dir, 'perspective_tf.jpg'), perspective_tf_frame)


# cv2.imshow('Harris Corner Detection', harris_frame)
# # cv2.imshow('Refined Harris Corner Detection', refined_harris_frame)
# cv2.imshow('Harris Corner Detection Rotated', harris_rotated_frame)
# cv2.imshow('Harris Corner Detection Scaled Up', harris_scaled_up_frame)
# cv2.imshow('Harris Corner Detection Scaled Down', harris_scaled_down_frame)
# cv2.imshow('Harris Corner Detection Affine Transformed', harris_affine_tf_frame)
# cv2.imshow('Harris Corner Detection Perspective Transformed', harris_perspective_tf_frame)

# cv2.imshow('SIFT Feature Detection', sift_frame)
# cv2.imshow('SIFT Feature Detection Rotated', sift_rotated_frame)
# cv2.imshow('SIFT Feature Detection Scaled Up', sift_scaled_up_frame)
# cv2.imshow('SIFT Feature Detection Scaled Down', sift_scaled_down_frame)
# cv2.imshow('SIFT Feature Detection Affine Transformed', sift_affine_tf_frame)
# cv2.imshow('SIFT Feature Detection Perspective Transformed', sift_perspective_tf_frame)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

