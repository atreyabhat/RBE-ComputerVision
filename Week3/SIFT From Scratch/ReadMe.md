# Scale Invariant Feature Transform (SIFT) Implementation

This repository contains a basic implementation of the Scale Invariant Feature Transform (SIFT) algorithm in Python using OpenCV, NumPy. The implementation focuses on  the first two steps of SIFT.

## Overview

The algorithm consists of following steps, including:

1. **Scale-Space Extrema Detection**: Detect potential keypoints by identifying local extrema in a scale-space representation of the image.
2. **Accurate Keypoint Localization**: Refine the locations and scales of potential keypoints for accuracy by fitting a 3D quadratic function to the local sample points.

## Requirements

- Python 3.x
- NumPy
- OpenCV
- Matplotlib

## Results


![UnityHall_keypoints](https://github.com/atreyabhat/RBE-ComputerVision/assets/39030188/0447f974-1bcc-4d4b-a464-677dd1e2b4b5)
![Lenna_keypoints](https://github.com/atreyabhat/RBE-ComputerVision/assets/39030188/a78452d3-0f40-4c33-9e30-35530f189387)
<img src="https://github.com/atreyabhat/RBE-ComputerVision/assets/39030188/9563a8ee-2242-4ab0-914f-77f5015a050d" alt="book_keypoints" width="400"/>
