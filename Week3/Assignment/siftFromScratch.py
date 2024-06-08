import cv2
import numpy as np
from scipy.ndimage import convolve

def sift_scale_space_extrema(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Compute the Difference of Gaussian (DoG) pyramid
    pyramid = compute_dog_pyramid(gray)
    
    # Find the extrema in the scale space
    extrema = find_extrema(pyramid)
    
    return extrema

def compute_dog_pyramid(image):
    # Define the number of octaves and scales per octave
    num_octaves = 4
    num_scales = 5
    
    # Create an empty list to store the pyramid
    pyramid = []
    
    # Generate the Gaussian pyramid
    for octave in range(num_octaves):
        octave_images = []
        for scale in range(num_scales):
            # Compute the scale factor for the current scale
            sigma = 1.6 * 2 ** (scale / num_scales)
            
            # Apply Gaussian smoothing to the image
            smoothed = cv2.GaussianBlur(image, (0, 0), sigma)
            
            # Append the smoothed image to the octave
            octave_images.append(smoothed)
        
        # Append the octave to the pyramid
        pyramid.append(octave_images)
        
        # Downsample the image for the next octave
        image = cv2.resize(image, (0, 0), fx=0.5, fy=0.5)
    
    # Compute the Difference of Gaussian (DoG) pyramid
    dog_pyramid = []
    for octave_images in pyramid:
        octave_dogs = []
        for i in range(len(octave_images) - 1):
            dog = octave_images[i + 1] - octave_images[i]
            octave_dogs.append(dog)
        dog_pyramid.append(octave_dogs)
    
    return dog_pyramid

def find_extrema(pyramid):
    # Define the threshold for extrema detection
    threshold = 0.03
    
    # Create an empty list to store the extrema
    extrema = []
    
    # Iterate over the pyramid
    for octave in range(len(pyramid)):
        octave_extrema = []
        for scale in range(1, len(pyramid[octave]) - 1):
            # Get the current scale and its neighbors
            current = pyramid[octave][scale]
            above = pyramid[octave][scale - 1]
            below = pyramid[octave][scale + 1]
            
            # Find the extrema in the current scale
            mask = np.logical_and(current > above, current > below)
            mask = np.logical_and(mask, np.abs(current) > threshold)
            keypoints = np.argwhere(mask)
            
            # Append the extrema to the octave
            octave_extrema.extend(keypoints)
        
        # Append the octave extrema to the overall extrema
        extrema.append(octave_extrema)
    
    return extrema