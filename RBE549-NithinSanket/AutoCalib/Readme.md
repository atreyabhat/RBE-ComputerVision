# Camera Calibration using Zhang's Method

## 1. Introduction
Camera calibration is essential for computer vision tasks involving 3D geometry. It involves estimating parameters such as the focal length, distortion coefficients, and principal point of a camera. Zhengyou Zhang's method, introduced in his seminal paper, provides an efficient way to perform robust camera calibration.

## 2. Data
The calibration target used is a checkerboard pattern printed on A4 paper, where each square measures 21.5mm.

<img src="https://github.com/atreyabhat/RBE-ComputerVision/assets/39030188/bc1b378e-99f3-4deb-ba5e-ec1191488804" width="400">


## 3. Initial Parameter Estimation
### 3.1 Solving for Approximate Camera Intrinsic Matrix \( K \)
The initial estimate of the camera intrinsic matrix \( K \) is obtained using OpenCV's `findChessboardCorners` function.

### 3.2 Estimate Approximate Camera Extrinsics \( R \) and \( t \)
The camera extrinsics \( R \) (rotation matrix) and \( t \) (translation vector) are estimated

### 3.3 Approximate Distortion Parameters \( k \)
Given minimal distortion assumptions, the initial estimate of distortion parameters \( k = [0, 0]^T \) is used.

## 4. Non-linear Geometric Error Minimization

### 4.1 Minimization Objective
The objective is to minimize the geometric error, which is defined as the sum of the differences between actual image points \( x_{i,j} \) and projected points \( \hat{x}_{i,j} \).

### 4.2 Optimization Process
Scipy's optimization methods are employed to iteratively adjust the camera intrinsic parameters \( fx, fy, cx, cy \) and distortion coefficients \( k1, k2 \).

## 5. Implementation
The implementation involves Python code using OpenCV for image processing and Scipy for optimization. The steps include:
- Loading calibration images
- Generating world points
- Finding chessboard corners
- Computing initial intrinsic matrix
- Computing extrinsic parameters
- Optimizing intrinsic matrix and distortion coefficients
- Computing reprojection errors
- Visualizing results and saving rectified images

<img src="https://github.com/atreyabhat/RBE-ComputerVision/assets/39030188/39aa3777-e2bd-4027-a524-9a81c1bfbdce" width="400">


## References
- Zhengyou Zhang, "A Flexible New Technique for Camera Calibration," [Link to paper](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/tr98-71.pdf)

## Usage
Run camCalib_Zhang.py
