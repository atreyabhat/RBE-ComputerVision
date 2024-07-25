import cv2
import numpy as np
import glob

# Number of inside corners in the checkerboard
num_corners_x = 9  # Number of corners along the width
num_corners_y = 7  # Number of corners along the height

# Termination criteria for the iterative algorithm
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Prepare object points (0,0,0), (1,0,0), (2,0,0), ....,(num_corners_x-1, num_corners_y-1, 0)
objp = np.zeros((num_corners_y*num_corners_x, 3), np.float32)
objp[:, :2] = np.mgrid[0:num_corners_x, 0:num_corners_y].T.reshape(-1, 2)

# Arrays to store object points and image points from all the images
objpoints = []  # 3d points in real world space
imgpoints = []  # 2d points in image plane

# Read images
path = 'calibration_data/data2/'
images = glob.glob(path + '*.jpg')

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (num_corners_x, num_corners_y), None)
    
    # If found, add object points, image points (after refining them)
    if ret:
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)
        
        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, (num_corners_x, num_corners_y), corners2, ret)
        cv2.imshow('img', img)
        cv2.waitKey(0)

cv2.destroyAllWindows()

# Calibration
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

# Save the calibration results
np.save(path + 'calibration_matrix.npy', mtx)
np.save(path + 'distortion_coefficients.npy', dist)

# Calculate the re-projection error
mean_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
    mean_error += error

mean_error /= len(objpoints)
np.save(path + 'reprojection_error.npy', mean_error)

print("Camera matrix:\n", mtx)
print("Distortion coefficients:\n", dist)
print("Re-projection error:\n", mean_error)

# Undistort example image
img = cv2.imread(images[10])
h, w = img.shape[:2]
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

# Undistort
dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

# Crop and save the image
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]
cv2.imwrite(path + 'calibresult.jpg', dst)
