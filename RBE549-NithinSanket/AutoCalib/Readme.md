# Camera Calibration using Zhang's Method

## 1. Introduction
Camera calibration is essential for computer vision tasks involving 3D geometry. It involves estimating parameters such as the focal length, distortion coefficients, and principal point of a camera. Zhengyou Zhang's method, introduced in his seminal paper, provides an efficient way to perform robust camera calibration.

## 2. Data
The calibration target used is a checkerboard pattern printed on A4 paper, where each square measures 21.5mm.

## 3. Initial Parameter Estimation
### 3.1 Solving for Approximate Camera Intrinsic Matrix \( K \)
The initial estimate of the camera intrinsic matrix \( K \) is obtained using OpenCV's `findChessboardCorners` function.

### 3.2 Estimate Approximate Camera Extrinsics \( R \) and \( t \)
The camera extrinsics \( R \) (rotation matrix) and \( t \) (translation vector) are estimated

### 3.3 Approximate Distortion Parameters \( k \)
Given minimal distortion assumptions, the initial estimate of distortion parameters \( k = [0, 0]^T \) is used.

## 4. Non-linear Geometric Error Minimization
### 4.1 Minimization Objective
The goal is to minimize the geometric error defined as:

\[ \sum_{i=1}^{N} \sum_{j=1}^{M} \| x_{i,j} - \hat{x}_{i,j}(K, R_i, t_i, X_j, k) \| \]

where \( x_{i,j} \) and \( \hat{x}_{i,j} \) represent image and projected points, respectively.

### 4.2 Optimization Process
Scipy's optimization methods are employed to minimize the above geometric error, refining the parameters \( fx, fy, cx, cy, k1, k2 \).

## References
- Zhengyou Zhang, "A Flexible New Technique for Camera Calibration," [Link to paper](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/tr98-71.pdf)


## Usage
Run camCalib_Zhang.py
