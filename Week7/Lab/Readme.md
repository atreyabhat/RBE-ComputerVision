# Epipolar Geometry

This project explores fundamental concepts in computer vision using OpenCV in Python. It covers topics such as stereo imaging, depth estimation, and epipolar geometry analysis.

## Parts Overview

### Part 1: Triangular Pyramid to Chessboard

Implement a tutorial to render a triangular pyramid onto a chessboard pattern.

### Part 2: Depth Map Generation from Stereo Images

Generate a depth map using stereo images to estimate the spatial distances of objects.

### Part 3: Epipolar Geometry Analysis
![epipolars](https://github.com/user-attachments/assets/ffa9ccdc-bfab-4856-a818-1769e429b9c8)



## Fundamental Matrix (F)

The fundamental matrix \( F \) relates corresponding points between two images. For points \( x \) in the first image and \( x' \) in the second image, the epipolar constraint \( x'^T F x = 0 \) holds true.

## Essential Matrix (E)

When camera intrinsic parameters \( K \) and \( K' \) are known, the essential matrix \( E \) relates to \( F \) through \( E = K'^T F K \).

## Epipolar Lines

- **First Image:** For a point \( x \), the corresponding epipolar line \( l' \) in the second image is \( l' = F x \).
- **Second Image:** For a point \( x' \), the corresponding epipolar line \( l \) in the first image is \( l = F^T x' \).

## Data

![globe_right](https://github.com/user-attachments/assets/534a3c72-b71a-47de-b9f1-2c974f758fd0)

![globe_left](https://github.com/user-attachments/assets/d3afa867-0526-4f67-8c37-cf9007140344)
![globe_center](https://github.com/user-attachments/assets/95dc9e8f-001d-45bd-a197-79ce8dd45b93)


## Results


![globe_lc](https://github.com/user-attachments/assets/1d20fb5f-d764-42c2-afab-acc7e4a8bce9)
![globe_cr](https://github.com/user-attachments/assets/33912466-2cbc-457b-b2b0-19d1a8d3866c)





