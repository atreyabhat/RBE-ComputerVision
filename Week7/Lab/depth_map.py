import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

# Load left and right images in grayscale
imgL = cv.imread('aloeL.jpg', cv.IMREAD_GRAYSCALE)
imgR = cv.imread('aloeR.jpg', cv.IMREAD_GRAYSCALE)

# Create a StereoBM object for stereo matching
stereo = cv.StereoBM_create()

# Set parameters for stereo matching
stereo.setNumDisparities(128)   # Number of disparities.  divisible by 16.
stereo.setBlockSize(15)         # Matched block size. odd number

stereo.setPreFilterSize(9)      # Size of the pre-filtering window (5-255). Larger values may reduce noise.
stereo.setPreFilterCap(31)      # Maximum allowed pre-filtered disparity value.

# stereo.setTextureThreshold(30)

stereo.setUniquenessRatio(15)   # Margin in percentage by which the best (minimum) computed cost function value should "win" the second best value.

stereo.setSpeckleRange(32)      # Maximum disparity variation within each connected component.
stereo.setSpeckleWindowSize(100) # Maximum size of smooth disparity regions to consider their noise speckles and invalidate.

# Compute the disparity map
disparity = stereo.compute(imgL, imgR)

# Save the disparity map as an image
cv.imwrite('disparity_map.png', disparity)

# Display the disparity map
plt.imshow(disparity, 'gray')
plt.colorbar()
plt.show()
