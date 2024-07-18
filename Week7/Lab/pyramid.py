import numpy as np
import cv2 as cv

# Function to draw a pyramid on an image
def draw_pyramid(img, corners, imgpts):
    imgpts = np.int32(imgpts).reshape(-1,2)
    
    # Draw base of the pyramid
    img = cv.drawContours(img, [imgpts[:4]], -1, (0,255,0), -3)
    
    # Draw sides of the pyramid
    for i in range(4):
        img = cv.line(img, tuple(imgpts[i]), tuple(imgpts[4]), (0,0,255), 3)
    
    return img

# Termination criteria for cornerSubPix
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Create a 3D model of a triangular pyramid
pyramid = np.float32([[0,0,0], [0,2,0], [2,2,0], [2,0,0], [1,1,-5]])

# Load calibration data (intrinsic and distortion matrices)
calib_mat = np.load('data/calibration_matrix.npy')       # Intrinsic matrix
distortion_mat = np.load('data/dist_coeff.npy')          # Distortion coefficients

# Load the image for calibration
img = cv.imread('data/calibresult.jpg')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# Define the known pattern (7x6 corners on a chessboard)
objp = np.zeros((6*7,3), np.float32)
objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)

# Find chessboard corners in the image
ret, corners = cv.findChessboardCorners(gray, (7,6), None)

if ret == True:
    # Refine corner locations using cornerSubPix
    corners2 = cv.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
    
    # Estimate the pose of the chessboard in 3D space (solvePnPRansac)
    _, R, C, _ = cv.solvePnPRansac(objp, corners2, calib_mat, distortion_mat)
    
    # Project the 3D points of the pyramid onto the image
    imgpts, _ = cv.projectPoints(pyramid, R, C, calib_mat, distortion_mat)
    
    # Draw the projected pyramid on the image
    img = draw_pyramid(img, corners2, imgpts)
    
    # Display the image with the pyramid
    cv.imshow('Pyramid projection', img)
    cv.waitKey(0)

# Clean up windows
cv.destroyAllWindows()

# Save the image with the pyramid drawn on it
cv.imwrite('triangular.jpg', img)
