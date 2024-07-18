# Depth Map and Epipolar Geometry

This project explores fundamental concepts in computer vision using OpenCV in Python. It covers topics such as stereo imaging, depth estimation, and epipolar geometry analysis.


### Part 1: Triangular Pyramid to Chessboard (pyramid.py)

Implement a tutorial to render a triangular pyramid onto a chessboard pattern.

### Part 2: Depth Map Generation from Stereo Images (depth_map.py)

Generate a depth map using stereo images to estimate the spatial distances of objects.

### Part 3: Epipolar Geometry Analysis (epipolar_geometry.py)

This part analyzes epipolar geometry using OpenCV, focusing on:
- **Fundamental Matrix (F)**: Relates corresponding points between two images.
- **Essential Matrix (E)**: Relates to F when camera intrinsic parameters \( K \) and \( K' \) are known.
- **Epipolar Lines**: Describes the relationship between points and lines in two images.

## Data

### Input Images:
![globe_right](https://github.com/user-attachments/assets/534a3c72-b71a-47de-b9f1-2c974f758fd0)
![globe_left](https://github.com/user-attachments/assets/d3afa867-0526-4f67-8c37-cf9007140344)
![globe_center](https://github.com/user-attachments/assets/95dc9e8f-001d-45bd-a197-79ce8dd45b93)

## Results

### Epipolar Geometry Results:
- Pair: Left and Center Images
  ![globe_lc](https://github.com/user-attachments/assets/1d20fb5f-d764-42c2-afab-acc7e4a8bce9)
- Pair: Center and Right Images
  ![globe_cr](https://github.com/user-attachments/assets/33912466-2cbc-457b-b2b0-19d1a8d3866c)

